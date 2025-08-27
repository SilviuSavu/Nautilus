#!/usr/bin/env python3
"""
Ultra-Fast ML Engine with Enhanced MessageBus Integration
FastAPI server with MessageBus background tasks for sub-5ms ML predictions.

Key Features:
- FastAPI REST endpoints (backward compatibility)
- Background MessageBus processing for ultra-low latency
- Neural Engine hardware acceleration
- Real-time prediction streaming
- Performance monitoring and optimization
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Import enhanced ML engine with MessageBus
from enhanced_ml_messagebus_integration import enhanced_ml_engine, MLPredictionResult
from ml_hardware_router import ml_hardware_router, route_ml_inference

# Import clock for deterministic operations
from clock import get_ml_clock

logger = logging.getLogger(__name__)


class UltraFastMLEngine:
    """
    Ultra-Fast ML Engine combining FastAPI REST endpoints with MessageBus
    
    Architecture:
    - FastAPI for external REST API (backward compatibility)
    - MessageBus for internal engine communication (sub-5ms)
    - Background tasks for continuous ML processing
    - Hardware-accelerated inference with Neural Engine
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Nautilus Ultra-Fast ML Engine",
            version="2.0.0",
            description="Enhanced ML Engine with MessageBus and Neural Engine acceleration",
            lifespan=self.lifespan
        )
        
        self.is_running = False
        self.start_time = time.time()
        self.clock = get_ml_clock()
        
        # Performance metrics
        self.api_requests_processed = 0
        self.messagebus_predictions_processed = 0
        self.neural_engine_predictions = 0
        self.average_api_response_time_ms = 0.0
        self.average_messagebus_latency_ms = 0.0
        
        # Background tasks
        self.background_tasks = []
        
        self.setup_routes()
        logger.info("🚀 Ultra-Fast ML Engine initialized")
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management with MessageBus integration"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
    
    async def start_engine(self) -> None:
        """Start Ultra-Fast ML Engine with MessageBus"""
        try:
            logger.info("🚀 Starting Ultra-Fast ML Engine with MessageBus...")
            
            # Initialize ML hardware router
            await ml_hardware_router.initialize_hardware_detection()
            
            # Initialize enhanced ML engine with MessageBus
            await enhanced_ml_engine.initialize()
            
            # Start background processing tasks
            self.background_tasks = [
                asyncio.create_task(self._continuous_prediction_processor()),
                asyncio.create_task(self._performance_optimizer()),
                asyncio.create_task(self._health_monitor())
            ]
            
            self.is_running = True
            logger.info("✅ Ultra-Fast ML Engine started successfully")
            logger.info("   🧠 Neural Engine: Ready for <5ms predictions")
            logger.info("   📡 MessageBus: Active for real-time communication")
            logger.info("   🔧 Hardware Router: Intelligent workload routing active")
            
        except Exception as e:
            logger.error(f"Failed to start Ultra-Fast ML Engine: {e}")
            raise
    
    async def stop_engine(self) -> None:
        """Stop Ultra-Fast ML Engine and cleanup"""
        logger.info("🔄 Stopping Ultra-Fast ML Engine...")
        self.is_running = False
        
        # Stop background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Stop enhanced ML engine
        await enhanced_ml_engine.stop()
        
        logger.info("✅ Ultra-Fast ML Engine stopped")
    
    def setup_routes(self) -> None:
        """Setup FastAPI routes for ML Engine"""
        
        @self.app.get("/health")
        async def health_check():
            """Enhanced health check with MessageBus status"""
            messagebus_health = await enhanced_ml_engine.get_performance_summary()
            
            return {
                "status": "healthy" if self.is_running else "stopped",
                "engine_type": "ultra_fast_ml",
                "version": "2.0.0",
                "uptime_seconds": time.time() - self.start_time,
                "api_requests_processed": self.api_requests_processed,
                "messagebus_predictions": self.messagebus_predictions_processed,
                "neural_engine_predictions": self.neural_engine_predictions,
                "average_api_response_ms": self.average_api_response_time_ms,
                "messagebus_active": enhanced_ml_engine.messagebus_client is not None,
                "neural_engine_available": enhanced_ml_engine.neural_engine_available,
                "hardware_router_active": ml_hardware_router.neural_engine_available,
                "performance_summary": messagebus_health
            }
        
        @self.app.get("/metrics")
        async def get_comprehensive_metrics():
            """Get comprehensive ML engine metrics"""
            
            # Get MessageBus performance metrics
            messagebus_metrics = await enhanced_ml_engine.get_performance_summary()
            
            # Get hardware routing statistics
            routing_stats = ml_hardware_router.get_routing_statistics()
            
            return {
                "api_performance": {
                    "requests_processed": self.api_requests_processed,
                    "average_response_time_ms": self.average_api_response_time_ms,
                    "requests_per_second": self.api_requests_processed / max(1, time.time() - self.start_time)
                },
                "messagebus_performance": messagebus_metrics,
                "hardware_routing": routing_stats,
                "system_status": {
                    "uptime_seconds": time.time() - self.start_time,
                    "engine_running": self.is_running,
                    "neural_engine_active": enhanced_ml_engine.neural_engine_available,
                    "messagebus_connected": enhanced_ml_engine.messagebus_client is not None
                }
            }
        
        @self.app.get("/models")
        async def get_loaded_models():
            """Get all loaded ML models"""
            models_info = []
            for model_id, model_info in enhanced_ml_engine.loaded_models.items():
                models_info.append({
                    "model_id": model_info.model_id,
                    "model_type": model_info.model_type,
                    "version": model_info.version,
                    "accuracy": model_info.accuracy,
                    "last_trained": model_info.last_trained.isoformat(),
                    "training_samples": model_info.training_samples,
                    "inference_time_ms": model_info.inference_time_ms,
                    "enabled": model_info.enabled
                })
            
            return {
                "models": models_info,
                "total_models": len(models_info),
                "enabled_models": len([m for m in models_info if m["enabled"]])
            }
        
        # ==================== ML PREDICTION ENDPOINTS ====================
        
        @self.app.post("/ml/predict/{model_id}")
        async def make_ml_prediction(model_id: str, input_data: Dict[str, Any], 
                                   background_tasks: BackgroundTasks):
            """Make ML prediction with hardware acceleration"""
            
            start_time = self.clock.timestamp()
            
            try:
                # Make prediction using enhanced ML engine
                prediction_result = await enhanced_ml_engine.make_prediction(
                    model_id, input_data
                )
                
                # Update API metrics
                api_response_time = (self.clock.timestamp() - start_time) * 1000
                self._update_api_metrics(api_response_time)
                
                # Add background task for additional processing if needed
                background_tasks.add_task(
                    self._post_prediction_processing, 
                    prediction_result
                )
                
                return {
                    "status": "success",
                    "prediction_id": prediction_result.prediction_id,
                    "model_id": model_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms,
                    "hardware_used": prediction_result.hardware_used,
                    "api_response_time_ms": api_response_time,
                    "timestamp": prediction_result.timestamp
                }
                
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/price")
        async def predict_price_movement(data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Predict price movement using optimized model"""
            
            start_time = self.clock.timestamp()
            
            try:
                symbol = data.get("symbol", "UNKNOWN")
                market_data = {
                    "price": data.get("current_price", 0),
                    "volume": data.get("volume", 0),
                    "timestamp": start_time
                }
                
                # Use enhanced ML engine for prediction
                prediction_result = await enhanced_ml_engine.predict_price_movement(symbol, market_data)
                
                # Update metrics
                api_response_time = (self.clock.timestamp() - start_time) * 1000
                self._update_api_metrics(api_response_time)
                
                # Background processing
                background_tasks.add_task(
                    self._post_prediction_processing,
                    prediction_result
                )
                
                return {
                    "prediction_type": "price_movement",
                    "symbol": symbol,
                    "prediction_id": prediction_result.prediction_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms,
                    "hardware_used": prediction_result.hardware_used,
                    "api_response_time_ms": api_response_time
                }
                
            except Exception as e:
                logger.error(f"Price prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/regime")
        async def detect_market_regime(data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Detect market regime using ML"""
            
            start_time = self.clock.timestamp()
            
            try:
                # Use enhanced ML engine
                prediction_result = await enhanced_ml_engine.predict_market_regime(data)
                
                # Update metrics
                api_response_time = (self.clock.timestamp() - start_time) * 1000
                self._update_api_metrics(api_response_time)
                
                # Background processing
                background_tasks.add_task(
                    self._post_prediction_processing,
                    prediction_result
                )
                
                return {
                    "prediction_type": "market_regime",
                    "prediction_id": prediction_result.prediction_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms,
                    "hardware_used": prediction_result.hardware_used,
                    "api_response_time_ms": api_response_time
                }
                
            except Exception as e:
                logger.error(f"Regime detection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/volatility")
        async def forecast_volatility(data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Forecast volatility using ML"""
            
            start_time = self.clock.timestamp()
            
            try:
                symbol = data.get("symbol", "UNKNOWN")
                historical_data = data.get("historical_prices", [])
                
                # Use enhanced ML engine
                prediction_result = await enhanced_ml_engine.forecast_volatility(symbol, historical_data)
                
                # Update metrics
                api_response_time = (self.clock.timestamp() - start_time) * 1000
                self._update_api_metrics(api_response_time)
                
                # Background processing
                background_tasks.add_task(
                    self._post_prediction_processing,
                    prediction_result
                )
                
                return {
                    "prediction_type": "volatility_forecast",
                    "symbol": symbol,
                    "prediction_id": prediction_result.prediction_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms,
                    "hardware_used": prediction_result.hardware_used,
                    "api_response_time_ms": api_response_time
                }
                
            except Exception as e:
                logger.error(f"Volatility forecasting failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/predict/risk")
        async def predict_portfolio_risk(data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Predict portfolio risk using ML"""
            
            start_time = self.clock.timestamp()
            
            try:
                # Use enhanced ML engine
                prediction_result = await enhanced_ml_engine.predict_portfolio_risk(data)
                
                # Update metrics
                api_response_time = (self.clock.timestamp() - start_time) * 1000
                self._update_api_metrics(api_response_time)
                
                # Background processing
                background_tasks.add_task(
                    self._post_prediction_processing,
                    prediction_result
                )
                
                return {
                    "prediction_type": "portfolio_risk",
                    "prediction_id": prediction_result.prediction_id,
                    "prediction": prediction_result.prediction,
                    "confidence": prediction_result.confidence,
                    "processing_time_ms": prediction_result.processing_time_ms,
                    "hardware_used": prediction_result.hardware_used,
                    "api_response_time_ms": api_response_time
                }
                
            except Exception as e:
                logger.error(f"Risk prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== MODEL MANAGEMENT ENDPOINTS ====================
        
        @self.app.post("/ml/train/{model_id}")
        async def train_model(model_id: str, training_data: Dict[str, Any], 
                            background_tasks: BackgroundTasks):
            """Train ML model with deterministic clock"""
            
            try:
                training_config = training_data.get("config", {})
                data = training_data.get("data", {})
                
                # Use enhanced ML engine for training
                training_result = await enhanced_ml_engine.train_model(model_id, data, training_config)
                
                # Background task for post-training optimization
                background_tasks.add_task(
                    self._post_training_processing,
                    model_id, training_result
                )
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "training_result": training_result
                }
                
            except Exception as e:
                logger.error(f"Model training failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/ml/models/{model_id}/update")
        async def update_model_weights(model_id: str, weight_updates: Dict[str, Any]):
            """Update model weights for online learning"""
            
            try:
                success = await enhanced_ml_engine.update_model_weights(model_id, weight_updates)
                
                return {
                    "status": "success" if success else "failed",
                    "model_id": model_id,
                    "updated": success
                }
                
            except Exception as e:
                logger.error(f"Model update failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ==================== HARDWARE ROUTING ENDPOINTS ====================
        
        @self.app.get("/ml/hardware/status")
        async def get_hardware_status():
            """Get ML hardware acceleration status"""
            
            routing_stats = ml_hardware_router.get_routing_statistics()
            
            return {
                "hardware_detection": {
                    "neural_engine_available": enhanced_ml_engine.neural_engine_available,
                    "metal_gpu_available": ml_hardware_router.metal_gpu_available,
                    "m4_max_detected": enhanced_ml_engine.m4_max_detected
                },
                "performance_metrics": {
                    "neural_engine_predictions": enhanced_ml_engine.neural_engine_predictions,
                    "cpu_fallback_predictions": enhanced_ml_engine.cpu_fallback_predictions,
                    "hardware_acceleration_ratio": enhanced_ml_engine.hardware_acceleration_ratio
                },
                "routing_statistics": routing_stats
            }
        
        @self.app.post("/ml/hardware/route-test")
        async def test_hardware_routing(test_data: Dict[str, Any]):
            """Test hardware routing decision"""
            
            try:
                model_id = test_data.get("model_id", "test_model")
                complexity = test_data.get("complexity", "medium")
                data_size = test_data.get("data_size", 1000)
                target_latency = test_data.get("target_latency_ms", 5.0)
                
                # Get routing decision
                routing_decision = await route_ml_inference(
                    model_id, complexity, data_size, target_latency
                )
                
                return {
                    "routing_decision": {
                        "primary_hardware": routing_decision.primary_hardware.value,
                        "fallback_hardware": [hw.value for hw in routing_decision.fallback_hardware],
                        "confidence": routing_decision.confidence,
                        "estimated_latency_ms": routing_decision.estimated_latency_ms,
                        "estimated_performance_gain": routing_decision.estimated_performance_gain,
                        "reasoning": routing_decision.reasoning
                    }
                }
                
            except Exception as e:
                logger.error(f"Hardware routing test failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # ==================== BACKGROUND PROCESSING ====================
    
    async def _continuous_prediction_processor(self) -> None:
        """Background task for continuous ML prediction processing"""
        while self.is_running:
            try:
                # Process any queued predictions from MessageBus
                # This would handle high-frequency prediction requests
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                logger.error(f"Continuous prediction processor error: {e}")
                await asyncio.sleep(1)
    
    async def _performance_optimizer(self) -> None:
        """Background task for performance optimization"""
        while self.is_running:
            try:
                # Optimize model performance and routing decisions
                # Based on actual performance metrics
                
                # Update hardware metrics if we have real performance data
                # This would analyze recent predictions and update routing preferences
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Performance optimizer error: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitor(self) -> None:
        """Background task for health monitoring"""
        while self.is_running:
            try:
                # Monitor engine health and performance
                if self.api_requests_processed > 0:
                    current_rps = self.api_requests_processed / (time.time() - self.start_time)
                    if current_rps > 100:  # High load
                        logger.info(f"🔥 High ML prediction load: {current_rps:.1f} RPS")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    def _update_api_metrics(self, response_time_ms: float) -> None:
        """Update API performance metrics"""
        self.api_requests_processed += 1
        
        # Update average response time
        self.average_api_response_time_ms = (
            (self.average_api_response_time_ms * (self.api_requests_processed - 1) + 
             response_time_ms) / self.api_requests_processed
        )
    
    async def _post_prediction_processing(self, prediction_result: MLPredictionResult) -> None:
        """Background processing after prediction"""
        try:
            # Update hardware metrics based on actual performance
            await ml_hardware_router.update_hardware_metrics(
                prediction_result.hardware_used.lower().replace(" ", "_"),
                prediction_result.processing_time_ms,
                True  # Assume success if we got here
            )
            
            # Track neural engine usage
            if prediction_result.hardware_used == "Neural Engine":
                self.neural_engine_predictions += 1
            
            logger.debug(f"Post-processing completed for prediction {prediction_result.prediction_id}")
            
        except Exception as e:
            logger.error(f"Post-prediction processing failed: {e}")
    
    async def _post_training_processing(self, model_id: str, training_result: Dict[str, Any]) -> None:
        """Background processing after model training"""
        try:
            # Optimize model performance after training
            logger.info(f"Post-training optimization for {model_id}: {training_result}")
            
        except Exception as e:
            logger.error(f"Post-training processing failed: {e}")


# Create Ultra-Fast ML Engine instance
ultra_fast_ml_engine = UltraFastMLEngine()

# Export the FastAPI app
app = ultra_fast_ml_engine.app


if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8400"))
    
    logger.info(f"🚀 Starting Ultra-Fast ML Engine on {host}:{port}")
    logger.info("   🧠 Features: Neural Engine + MessageBus + Hardware Routing")
    logger.info("   ⚡ Target: <5ms ML predictions with sub-5ms MessageBus latency")
    
    # Start FastAPI server with lifespan management
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
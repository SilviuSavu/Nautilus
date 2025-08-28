#!/usr/bin/env python3
"""
Enhanced ML Engine with Engine Awareness
Example of how to integrate the new engine awareness system with an existing engine.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Import existing ML engine components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the new engine awareness system
from engines.common.nautilus_environment import get_nautilus_environment
from engines.common.engine_identity import (
    EngineIdentity, EngineRole, ProcessingCapability, EngineCapabilities,
    DataSchema, DataFormat, PartnershipPreference
)
from engines.common.engine_discovery import EngineDiscoveryProtocol
from engines.common.intelligent_router import MessageRouter, TaskPriority, WorkflowTemplates
from engines.common.partnership_manager import PartnershipManager, PartnershipType


logger = logging.getLogger(__name__)


def create_enhanced_ml_engine_identity() -> EngineIdentity:
    """Create enhanced identity for ML Engine with full awareness capabilities"""
    
    capabilities = EngineCapabilities(
        supported_roles=[
            EngineRole.MACHINE_LEARNING,
            EngineRole.DATA_PROCESSOR,
            EngineRole.ANALYTICS
        ],
        processing_capabilities=[
            ProcessingCapability.MACHINE_LEARNING_INFERENCE,
            ProcessingCapability.REAL_TIME_STREAMING,
            ProcessingCapability.FEATURE_ENGINEERING,
            ProcessingCapability.BATCH_PROCESSING,
            ProcessingCapability.VOLATILITY_MODELING
        ],
        input_data_schemas=[
            DataSchema(
                name="market_features",
                format=DataFormat.PANDAS_DATAFRAME,
                required_fields=["timestamp", "symbol", "price", "volume"],
                optional_fields=["bid", "ask", "spread", "volatility"],
                description="Market data features for ML inference",
                example={
                    "timestamp": "2025-01-01T12:00:00Z",
                    "symbol": "AAPL",
                    "price": 150.25,
                    "volume": 1000000
                }
            ),
            DataSchema(
                name="engineered_features",
                format=DataFormat.NUMPY_ARRAY,
                required_fields=["feature_vector", "timestamp"],
                description="Pre-engineered features from Features Engine"
            ),
            DataSchema(
                name="factor_scores",
                format=DataFormat.JSON,
                required_fields=["symbol", "scores", "timestamp"],
                description="Factor scores from Factor Engine"
            )
        ],
        output_data_schemas=[
            DataSchema(
                name="price_predictions",
                format=DataFormat.JSON,
                required_fields=["symbol", "predicted_price", "confidence", "timestamp", "horizon_minutes"],
                optional_fields=["prediction_interval", "model_version"],
                description="Price predictions with confidence intervals"
            ),
            DataSchema(
                name="volatility_forecast",
                format=DataFormat.JSON,
                required_fields=["symbol", "volatility", "timestamp"],
                description="Volatility forecasts"
            ),
            DataSchema(
                name="regime_classification",
                format=DataFormat.JSON,
                required_fields=["market_regime", "confidence", "timestamp"],
                description="Market regime classification (bull/bear/sideways)"
            )
        ],
        partnership_preferences=[
            # Primary partnerships - critical for operation
            PartnershipPreference(
                engine_id="FEATURES_ENGINE",
                relationship_type="primary",
                data_flow_direction="input",
                preferred_message_types=["FEATURE_CALCULATION", "ENGINEERED_FEATURES"],
                latency_requirement_ms=5.0,
                reliability_requirement_pct=99.9,
                description="Primary source of engineered features for ML models"
            ),
            PartnershipPreference(
                engine_id="FACTOR_ENGINE", 
                relationship_type="primary",
                data_flow_direction="input",
                preferred_message_types=["FACTOR_CALCULATION"],
                latency_requirement_ms=10.0,
                reliability_requirement_pct=99.5,
                description="Source of fundamental and technical factors"
            ),
            
            # Secondary partnerships - important but not critical
            PartnershipPreference(
                engine_id="ANALYTICS_ENGINE",
                relationship_type="secondary", 
                data_flow_direction="output",
                preferred_message_types=["ML_PREDICTION", "ANALYTICS_RESULT"],
                latency_requirement_ms=20.0,
                reliability_requirement_pct=99.0,
                description="Send predictions for further analytics processing"
            ),
            PartnershipPreference(
                engine_id="STRATEGY_ENGINE",
                relationship_type="secondary",
                data_flow_direction="output", 
                preferred_message_types=["ML_PREDICTION", "STRATEGY_SIGNAL"],
                latency_requirement_ms=15.0,
                reliability_requirement_pct=99.0,
                description="Provide predictions for strategy generation"
            ),
            PartnershipPreference(
                engine_id="RISK_ENGINE",
                relationship_type="secondary",
                data_flow_direction="bidirectional",
                preferred_message_types=["ML_PREDICTION", "RISK_METRIC"],
                latency_requirement_ms=25.0,
                reliability_requirement_pct=98.0,
                description="Exchange predictions and risk assessments"
            ),
            
            # Optional partnerships - enhance capabilities
            PartnershipPreference(
                engine_id="WEBSOCKET_THGNN_ENGINE",
                relationship_type="optional",
                data_flow_direction="bidirectional",
                preferred_message_types=["ML_PREDICTION", "HFT_SIGNAL"],
                latency_requirement_ms=2.0,
                reliability_requirement_pct=95.0,
                description="Collaborate on high-frequency trading predictions"
            ),
            PartnershipPreference(
                engine_id="QUANTUM_PORTFOLIO_ENGINE",
                relationship_type="optional",
                data_flow_direction="output",
                preferred_message_types=["ML_PREDICTION"],
                latency_requirement_ms=50.0,
                reliability_requirement_pct=90.0,
                description="Provide predictions for quantum portfolio optimization"
            )
        ],
        hardware_requirements=[
            "neural_engine",
            "metal_gpu", 
            "unified_memory",
            "performance_cores"
        ],
        software_dependencies=[
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0", 
            "scikit-learn>=1.3.0",
            "mlx>=0.6.0"  # Apple MLX for M4 Max acceleration
        ]
    )
    
    return EngineIdentity(
        engine_id="ML_ENGINE",
        engine_name="Enhanced ML Engine with Awareness",
        engine_port=8400,
        capabilities=capabilities,
        version="2025.1.0"
    )


class EnhancedMLEngine:
    """Enhanced ML Engine with full engine awareness capabilities"""
    
    def __init__(self):
        # Core ML functionality (simplified)
        self.models = {
            "price_prediction": None,
            "volatility_forecast": None,
            "regime_classification": None
        }
        self.model_metrics = {
            "predictions_made": 0,
            "accuracy_score": 0.85,
            "last_training": datetime.now().isoformat()
        }
        
        # Engine awareness components
        self.identity = create_enhanced_ml_engine_identity()
        self.discovery_protocol: Optional[EngineDiscoveryProtocol] = None
        self.message_router: Optional[MessageRouter] = None
        self.partnership_manager: Optional[PartnershipManager] = None
        
        # Partnership tracking
        self.active_collaborations = {}
        self.collaboration_history = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "partner_interactions": {}
        }
        
        # Real-time data feeds from partners
        self.live_features: Dict[str, Any] = {}
        self.live_factors: Dict[str, Any] = {}
        
        self._running = False
    
    async def initialize(self):
        """Initialize the enhanced ML engine"""
        try:
            logger.info("Initializing Enhanced ML Engine...")
            
            # Initialize awareness components
            self.discovery_protocol = EngineDiscoveryProtocol(self.identity)
            await self.discovery_protocol.initialize()
            
            self.message_router = MessageRouter(self.identity, self.discovery_protocol)
            
            self.partnership_manager = PartnershipManager(
                self.identity,
                self.discovery_protocol,
                self.message_router
            )
            
            # Register event handlers for partnership events
            self._register_partnership_handlers()
            
            # Load ML models (simulated)
            await self._load_models()
            
            # Update engine health
            self.identity.update_health_status()
            
            logger.info("Enhanced ML Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced ML Engine: {e}")
            raise
    
    async def start(self):
        """Start the enhanced ML engine"""
        if self._running:
            return
            
        self._running = True
        
        # Start discovery protocol
        await self.discovery_protocol.start()
        
        # Start background tasks
        asyncio.create_task(self._collaboration_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Auto-establish partnerships
        await self._establish_initial_partnerships()
        
        logger.info("Enhanced ML Engine started and ready for collaboration")
    
    async def stop(self):
        """Stop the enhanced ML engine"""
        if not self._running:
            return
            
        self._running = False
        
        if self.discovery_protocol:
            await self.discovery_protocol.stop()
        
        logger.info("Enhanced ML Engine stopped")
    
    def _register_partnership_handlers(self):
        """Register handlers for partnership events"""
        from engines.common.engine_discovery import DiscoveryEventType
        
        async def handle_new_partner(event_data):
            """Handle discovery of potential partner"""
            partner_data = event_data["payload"]
            partner_id = partner_data["engine_id"]
            
            # Check if this is a preferred partner
            preferred_partners = self.identity.get_preferred_partners()
            
            if partner_id in preferred_partners:
                logger.info(f"Preferred partner discovered: {partner_id}")
                
                # Determine partnership type
                partnership_pref = next(
                    (p for p in self.identity.capabilities.partnership_preferences 
                     if p.engine_id == partner_id), None
                )
                
                if partnership_pref:
                    partnership_type = PartnershipType(partnership_pref.relationship_type)
                    
                    # Create partnership
                    partnership = self.partnership_manager.create_partnership(
                        partner_id,
                        partnership_type,
                        partnership_pref.latency_requirement_ms,
                        reliability_requirement=partnership_pref.reliability_requirement_pct / 100
                    )
                    
                    logger.info(f"Established {partnership_type.value} partnership with {partner_id}")
                    
                    # Start collaboration
                    await self._initiate_collaboration(partner_id, partnership_pref)
        
        self.discovery_protocol.register_event_handler(
            DiscoveryEventType.ENGINE_ANNOUNCEMENT,
            handle_new_partner
        )
    
    async def _load_models(self):
        """Load ML models (simulated)"""
        # Simulate model loading
        await asyncio.sleep(0.1)
        
        self.models = {
            "price_prediction": {
                "model_type": "LSTM",
                "accuracy": 0.85,
                "last_trained": datetime.now().isoformat(),
                "features_used": ["price", "volume", "technical_indicators"]
            },
            "volatility_forecast": {
                "model_type": "GARCH",
                "accuracy": 0.78,
                "last_trained": datetime.now().isoformat(),
                "features_used": ["returns", "volume", "volatility_surface"]
            },
            "regime_classification": {
                "model_type": "Hidden_Markov",
                "accuracy": 0.72,
                "last_trained": datetime.now().isoformat(),
                "features_used": ["market_indicators", "economic_factors"]
            }
        }
        
        logger.info("ML models loaded successfully")
    
    async def _establish_initial_partnerships(self):
        """Establish initial partnerships with available engines"""
        await asyncio.sleep(5)  # Wait for other engines to announce
        
        # Find and connect to preferred partners
        online_engines = self.discovery_protocol.get_online_engines()
        preferred_partners = self.identity.get_preferred_partners()
        
        for partner_id in preferred_partners:
            if partner_id in online_engines and partner_id not in self.partnership_manager.partnerships:
                
                partnership_pref = next(
                    (p for p in self.identity.capabilities.partnership_preferences 
                     if p.engine_id == partner_id), None
                )
                
                if partnership_pref:
                    partnership_type = PartnershipType(partnership_pref.relationship_type)
                    
                    partnership = self.partnership_manager.create_partnership(
                        partner_id,
                        partnership_type,
                        partnership_pref.latency_requirement_ms,
                        reliability_requirement=partnership_pref.reliability_requirement_pct / 100
                    )
                    
                    logger.info(f"Auto-established partnership with {partner_id}")
    
    async def _initiate_collaboration(self, partner_id: str, partnership_pref: PartnershipPreference):
        """Initiate collaboration with a partner engine"""
        
        if partner_id == "FEATURES_ENGINE":
            # Request real-time features
            await self._request_live_features(partner_id)
            
        elif partner_id == "FACTOR_ENGINE":
            # Request factor calculations
            await self._request_factor_scores(partner_id)
            
        elif partner_id == "ANALYTICS_ENGINE":
            # Offer to provide predictions
            await self._offer_predictions(partner_id)
            
        # Track collaboration
        self.active_collaborations[partner_id] = {
            "started_at": datetime.now().isoformat(),
            "collaboration_type": partnership_pref.data_flow_direction,
            "message_types": partnership_pref.preferred_message_types,
            "interactions": 0
        }
    
    async def _request_live_features(self, features_engine_id: str):
        """Request live features from Features Engine"""
        # This would send a request via the message router
        logger.info(f"Requesting live features from {features_engine_id}")
        
        # Simulate receiving features
        self.live_features = {
            "timestamp": datetime.now().isoformat(),
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "features": {
                "technical_indicators": {
                    "rsi": [45.2, 67.8, 52.1],
                    "macd": [0.15, -0.23, 0.08],
                    "bollinger_position": [0.3, 0.8, 0.4]
                },
                "market_microstructure": {
                    "bid_ask_spread": [0.01, 0.02, 0.015],
                    "order_flow_imbalance": [0.2, -0.1, 0.05]
                }
            }
        }
    
    async def _request_factor_scores(self, factor_engine_id: str):
        """Request factor scores from Factor Engine"""
        logger.info(f"Requesting factor scores from {factor_engine_id}")
        
        # Simulate receiving factor scores
        self.live_factors = {
            "timestamp": datetime.now().isoformat(),
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "factors": {
                "momentum": [0.65, -0.23, 0.41],
                "value": [-0.15, 0.78, 0.22],
                "quality": [0.82, 0.91, 0.67],
                "volatility": [0.34, 0.56, 0.28]
            }
        }
    
    async def _offer_predictions(self, analytics_engine_id: str):
        """Offer predictions to Analytics Engine"""
        logger.info(f"Offering predictions to {analytics_engine_id}")
        
        # This would establish a data flow to send predictions
        # For now, we'll just log the intent
    
    async def _collaboration_loop(self):
        """Background loop for maintaining collaborations"""
        while self._running:
            try:
                # Update live data from partners
                if self.live_features and self.live_factors:
                    # Generate predictions using collaborative data
                    predictions = await self._generate_collaborative_predictions()
                    
                    # Send predictions to interested partners
                    await self._broadcast_predictions(predictions)
                    
                    # Update performance metrics
                    self.model_metrics["predictions_made"] += len(predictions.get("symbols", []))
                
                # Update partnership performance
                await self._update_partnership_metrics()
                
                await asyncio.sleep(10)  # Collaborate every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in collaboration loop: {e}")
                await asyncio.sleep(30)
    
    async def _generate_collaborative_predictions(self) -> Dict[str, Any]:
        """Generate predictions using data from partner engines"""
        
        # Simulate ML inference using features and factors
        predictions = {
            "timestamp": datetime.now().isoformat(),
            "model_version": "collaborative_v1.0",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "predictions": {
                "price_targets": [155.30, 2750.80, 378.90],
                "confidence": [0.87, 0.79, 0.82],
                "horizon_minutes": 60,
                "volatility_forecast": [0.25, 0.31, 0.22],
                "regime_probability": {
                    "bull": [0.65, 0.78, 0.71],
                    "bear": [0.20, 0.12, 0.18],
                    "sideways": [0.15, 0.10, 0.11]
                }
            },
            "data_sources": {
                "features_from": "FEATURES_ENGINE",
                "factors_from": "FACTOR_ENGINE",
                "enhanced_by_collaboration": True
            }
        }
        
        # Update identity performance metrics
        response_time = 5.2  # Simulated processing time
        self.identity.update_performance_metrics(response_time, len(predictions["symbols"]))
        
        return predictions
    
    async def _broadcast_predictions(self, predictions: Dict[str, Any]):
        """Broadcast predictions to interested partner engines"""
        
        interested_partners = ["ANALYTICS_ENGINE", "STRATEGY_ENGINE", "RISK_ENGINE"]
        
        for partner_id in interested_partners:
            if partner_id in self.partnership_manager.partnerships:
                partnership = self.partnership_manager.partnerships[partner_id]
                
                # Update partnership performance (simulated successful send)
                self.partnership_manager.update_partnership_performance(
                    partner_id, 
                    latency_ms=8.5,  # Simulated network latency
                    success=True
                )
                
                # Track interaction
                if partner_id in self.active_collaborations:
                    self.active_collaborations[partner_id]["interactions"] += 1
                
                logger.debug(f"Sent predictions to {partner_id}")
    
    async def _update_partnership_metrics(self):
        """Update metrics for all partnerships"""
        
        for partner_id, collaboration in self.active_collaborations.items():
            if partner_id not in self.performance_metrics["partner_interactions"]:
                self.performance_metrics["partner_interactions"][partner_id] = {
                    "total_interactions": 0,
                    "successful_interactions": 0,
                    "average_latency_ms": 0.0
                }
            
            metrics = self.performance_metrics["partner_interactions"][partner_id]
            metrics["total_interactions"] = collaboration["interactions"]
            metrics["successful_interactions"] = collaboration["interactions"]  # Assume all successful for demo
            metrics["average_latency_ms"] = 7.8  # Simulated
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring"""
        while self._running:
            try:
                # Update health status
                self.identity.update_health_status()
                
                # Update performance metrics
                self.performance_metrics.update({
                    "total_requests": self.model_metrics["predictions_made"],
                    "successful_requests": self.model_metrics["predictions_made"],  # Assume all successful
                    "average_response_time": 5.2,  # Simulated
                    "model_accuracy": self.model_metrics["accuracy_score"],
                    "active_partnerships": len(self.partnership_manager.get_active_partnerships()),
                    "collaboration_score": self._calculate_collaboration_score()
                })
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(120)
    
    def _calculate_collaboration_score(self) -> float:
        """Calculate overall collaboration effectiveness score"""
        if not self.active_collaborations:
            return 0.0
        
        total_score = 0.0
        for partner_id, collaboration in self.active_collaborations.items():
            partnership = self.partnership_manager.partnerships.get(partner_id)
            if partnership:
                # Factor in partnership performance and relationship strength
                score = (partnership.performance_score + partnership.relationship_strength) / 2
                total_score += score
        
        return total_score / len(self.active_collaborations)
    
    # API Methods
    def get_health(self) -> Dict[str, Any]:
        """Get engine health status"""
        return {
            "status": "healthy",
            "engine_id": self.identity.engine_id,
            "version": self.identity.version,
            "uptime_seconds": self.identity.health.uptime_seconds,
            "performance_metrics": self.performance_metrics,
            "model_metrics": self.model_metrics,
            "active_partnerships": len(self.partnership_manager.get_active_partnerships()) if self.partnership_manager else 0,
            "collaboration_score": self._calculate_collaboration_score(),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_partnerships(self) -> Dict[str, Any]:
        """Get partnership information"""
        if not self.partnership_manager:
            return {"partnerships": []}
        
        return {
            "partnerships": [
                {
                    "partner_id": partner_id,
                    "type": partnership.partnership_type.value,
                    "status": partnership.status.value,
                    "performance_score": partnership.performance_score,
                    "relationship_strength": partnership.relationship_strength,
                    "message_count": partnership.message_count
                }
                for partner_id, partnership in self.partnership_manager.partnerships.items()
            ],
            "collaboration_history": self.collaboration_history[-10:],  # Last 10 events
            "active_collaborations": len(self.active_collaborations)
        }
    
    async def predict(self, symbols: List[str], horizon_minutes: int = 60) -> Dict[str, Any]:
        """Make predictions using collaborative data"""
        
        # Check if we have collaborative data
        if not self.live_features or not self.live_factors:
            logger.warning("Limited collaborative data available, using local models only")
        
        # Generate predictions
        predictions = await self._generate_collaborative_predictions()
        
        # Filter for requested symbols
        if symbols:
            # In a real implementation, filter predictions for requested symbols
            pass
        
        return predictions


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    engine = EnhancedMLEngine()
    await engine.initialize()
    await engine.start()
    
    app.state.engine = engine
    
    yield
    
    # Shutdown
    await engine.stop()


app = FastAPI(
    title="Enhanced ML Engine with Awareness",
    description="ML Engine with full engine awareness and collaboration capabilities",
    version="2025.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    engine = app.state.engine
    return engine.get_health()


@app.get("/api/v1/partnerships")
async def get_partnerships():
    """Get partnership information"""
    engine = app.state.engine
    return engine.get_partnerships()


@app.post("/api/v1/predict")
async def make_prediction(symbols: List[str], horizon_minutes: int = 60):
    """Make ML predictions"""
    engine = app.state.engine
    return await engine.predict(symbols, horizon_minutes)


@app.get("/api/v1/identity")
async def get_identity():
    """Get engine identity and capabilities"""
    engine = app.state.engine
    return engine.identity.to_dict()


@app.get("/api/v1/collaborations")
async def get_active_collaborations():
    """Get active collaborations"""
    engine = app.state.engine
    return {
        "active_collaborations": engine.active_collaborations,
        "performance_metrics": engine.performance_metrics,
        "live_data_status": {
            "features_available": bool(engine.live_features),
            "factors_available": bool(engine.live_factors),
            "last_feature_update": engine.live_features.get("timestamp") if engine.live_features else None,
            "last_factor_update": engine.live_factors.get("timestamp") if engine.live_factors else None
        }
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8400,
        log_level="info"
    )
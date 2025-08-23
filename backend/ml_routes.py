"""
ML API Routes for Nautilus Trading Platform
Provides RESTful endpoints for all ML capabilities including regime detection,
feature engineering, risk prediction, and real-time inference.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

from ml.config import MLConfig
from ml.market_regime import MarketRegimeDetector, RegimeType
from ml.feature_engineering import FeatureEngineer
from ml.model_lifecycle import ModelLifecycleManager
from ml.risk_prediction import RiskPredictor
from ml.inference_engine import InferenceEngine, InferenceRequest, ModelServer

logger = logging.getLogger(__name__)

# Initialize ML components
ml_config = MLConfig()
regime_detector = MarketRegimeDetector(ml_config)
feature_engineer = FeatureEngineer(ml_config)
lifecycle_manager = ModelLifecycleManager(ml_config)
risk_predictor = RiskPredictor()
inference_engine = InferenceEngine(ml_config)

router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])

# Health and Status Endpoints
@router.get("/health")
async def ml_health_check():
    """Check health status of all ML components."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {
                "regime_detector": await regime_detector.health_check(),
                "feature_engineer": await feature_engineer.health_check(),
                "lifecycle_manager": await lifecycle_manager.health_check(),
                "risk_predictor": await risk_predictor.health_check(),
                "inference_engine": await inference_engine.health_check(),
            },
            "models_loaded": len(inference_engine.model_servers),
            "active_predictions": inference_engine.get_active_requests_count()
        }
        
        # Check if any component is unhealthy
        unhealthy_components = [
            name for name, status in health_status["components"].items() 
            if not status.get("healthy", False)
        ]
        
        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
            
        return health_status
        
    except Exception as e:
        logger.error(f"ML health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"ML services unavailable: {str(e)}")

@router.get("/status")
async def ml_system_status():
    """Get comprehensive ML system status and metrics."""
    try:
        return {
            "system_metrics": await inference_engine.get_system_metrics(),
            "model_performance": await lifecycle_manager.get_model_performance_summary(),
            "regime_analysis": await regime_detector.get_current_regime_analysis(),
            "feature_stats": await feature_engineer.get_feature_statistics(),
            "risk_metrics": await risk_predictor.get_current_risk_metrics()
        }
    except Exception as e:
        logger.error(f"Error getting ML system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market Regime Detection Endpoints
@router.get("/regime/current")
async def get_current_regime():
    """Get current market regime prediction with confidence."""
    try:
        regime_state = await regime_detector.get_current_regime()
        return {
            "regime": regime_state.regime.value,
            "confidence": regime_state.confidence,
            "probabilities": regime_state.probabilities,
            "timestamp": regime_state.timestamp,
            "features_used": regime_state.features_used
        }
    except Exception as e:
        logger.error(f"Error getting current regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regime/predict")
async def predict_regime(features: Dict[str, float]):
    """Predict market regime from custom feature set."""
    try:
        regime_state = await regime_detector.predict_regime(features)
        return {
            "regime": regime_state.regime.value,
            "confidence": regime_state.confidence,
            "probabilities": regime_state.probabilities,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error predicting regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regime/history")
async def get_regime_history(
    days: int = Query(30, description="Number of days to retrieve"),
    include_probabilities: bool = Query(False, description="Include probability distributions")
):
    """Get historical regime predictions."""
    try:
        history = await regime_detector.get_regime_history(
            start_date=datetime.utcnow() - timedelta(days=days),
            end_date=datetime.utcnow(),
            include_probabilities=include_probabilities
        )
        return history
    except Exception as e:
        logger.error(f"Error getting regime history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feature Engineering Endpoints
@router.post("/features/compute")
async def compute_features(
    symbol: str,
    lookback_days: Optional[int] = 30,
    feature_groups: Optional[List[str]] = None
):
    """Compute features for a specific symbol."""
    try:
        features = await feature_engineer.compute_features(
            symbol=symbol,
            lookback_days=lookback_days,
            feature_groups=feature_groups
        )
        return {
            "symbol": symbol,
            "timestamp": features.timestamp,
            "features": features.features,
            "feature_groups": list(features.features.keys()),
            "total_features": sum(len(group_features) for group_features in features.features.values())
        }
    except Exception as e:
        logger.error(f"Error computing features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/correlation")
async def get_correlation_analysis(
    symbols: List[str] = Query(..., description="List of symbols to analyze"),
    lookback_days: int = Query(30, description="Lookback period for correlation"),
    method: str = Query("pearson", description="Correlation method")
):
    """Get multi-asset correlation analysis."""
    try:
        correlation_result = await feature_engineer.correlation_analyzer.analyze_cross_asset_correlation(
            symbols=symbols,
            lookback_days=lookback_days,
            method=method
        )
        return {
            "correlation_matrix": correlation_result.correlation_matrix.to_dict(),
            "clusters": correlation_result.clusters,
            "eigenvalues": correlation_result.eigenvalues,
            "timestamp": correlation_result.timestamp
        }
    except Exception as e:
        logger.error(f"Error computing correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Lifecycle Management Endpoints
@router.post("/models/retrain")
async def trigger_model_retraining(
    background_tasks: BackgroundTasks,
    model_type: str,
    force: bool = Query(False, description="Force retraining even if not needed")
):
    """Trigger model retraining process."""
    try:
        background_tasks.add_task(
            lifecycle_manager.trigger_retraining,
            model_type=model_type,
            force=force
        )
        return {
            "status": "retraining_initiated",
            "model_type": model_type,
            "timestamp": datetime.utcnow(),
            "message": f"Model retraining for {model_type} has been queued"
        }
    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/drift")
async def check_model_drift(
    model_type: str,
    threshold: float = Query(0.1, description="Drift detection threshold")
):
    """Check for model drift on specified model."""
    try:
        drift_result = await lifecycle_manager.check_model_drift(
            model_type=model_type,
            threshold=threshold
        )
        return {
            "model_type": model_type,
            "drift_detected": drift_result.drift_detected,
            "drift_score": drift_result.drift_score,
            "drift_types": drift_result.drift_types,
            "recommendation": drift_result.recommendation,
            "timestamp": drift_result.timestamp
        }
    except Exception as e:
        logger.error(f"Error checking model drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/performance")
async def get_model_performance(model_type: Optional[str] = None):
    """Get model performance metrics."""
    try:
        performance = await lifecycle_manager.get_model_performance(model_type)
        return performance
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Prediction Endpoints
@router.post("/risk/portfolio/optimize")
async def optimize_portfolio(
    holdings: Dict[str, float],
    method: str = Query("mean_variance", description="Optimization method"),
    risk_tolerance: float = Query(0.1, description="Risk tolerance parameter")
):
    """Optimize portfolio allocation using ML-enhanced methods."""
    try:
        optimization_result = await risk_predictor.optimize_portfolio(
            holdings=holdings,
            method=method,
            risk_tolerance=risk_tolerance
        )
        return {
            "optimal_weights": optimization_result.optimal_weights,
            "expected_return": optimization_result.expected_return,
            "expected_volatility": optimization_result.expected_volatility,
            "sharpe_ratio": optimization_result.sharpe_ratio,
            "method_used": method,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/var/calculate")
async def calculate_var(
    portfolio: Dict[str, float],
    confidence_level: float = Query(0.95, description="VaR confidence level"),
    horizon_days: int = Query(1, description="Risk horizon in days"),
    method: str = Query("monte_carlo", description="VaR calculation method")
):
    """Calculate Value at Risk using ML-enhanced methods."""
    try:
        var_result = await risk_predictor.calculate_var(
            portfolio=portfolio,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            method=method
        )
        return {
            "var": var_result.var,
            "expected_shortfall": var_result.expected_shortfall,
            "confidence_level": confidence_level,
            "horizon_days": horizon_days,
            "method": method,
            "scenario_analysis": var_result.scenario_analysis,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/stress-test")
async def run_stress_test(
    portfolio: Dict[str, float],
    scenarios: Optional[List[str]] = None,
    custom_shocks: Optional[Dict[str, float]] = None
):
    """Run stress testing scenarios on portfolio."""
    try:
        stress_result = await risk_predictor.run_stress_test(
            portfolio=portfolio,
            scenarios=scenarios,
            custom_shocks=custom_shocks
        )
        return {
            "stress_test_results": stress_result.scenario_results,
            "worst_case_loss": stress_result.worst_case_loss,
            "risk_metrics": stress_result.risk_metrics,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time Inference Endpoints
@router.post("/inference/predict")
async def make_prediction(request: InferenceRequest):
    """Make real-time ML prediction."""
    try:
        result = await inference_engine.predict(request)
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_used": result.model_name,
            "inference_time_ms": result.inference_time_ms,
            "timestamp": result.timestamp,
            "request_id": result.request_id
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inference/models")
async def list_available_models():
    """List all available models for inference."""
    try:
        models = await inference_engine.list_available_models()
        return {
            "available_models": models,
            "total_models": len(models),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inference/metrics")
async def get_inference_metrics():
    """Get real-time inference metrics and performance data."""
    try:
        metrics = await inference_engine.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting inference metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inference/models/load")
async def load_model(
    model_name: str,
    model_type: str,
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Load a new model into the inference engine."""
    try:
        success = await inference_engine.load_model(
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            config=config or {}
        )
        return {
            "success": success,
            "model_name": model_name,
            "model_type": model_type,
            "timestamp": datetime.utcnow(),
            "message": f"Model {model_name} loaded successfully" if success else f"Failed to load model {model_name}"
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/inference/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from the inference engine."""
    try:
        success = await inference_engine.unload_model(model_name)
        return {
            "success": success,
            "model_name": model_name,
            "timestamp": datetime.utcnow(),
            "message": f"Model {model_name} unloaded successfully" if success else f"Failed to unload model {model_name}"
        }
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring and Analytics Endpoints
@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive ML monitoring dashboard data."""
    try:
        dashboard_data = {
            "system_status": await ml_health_check(),
            "inference_metrics": await inference_engine.get_system_metrics(),
            "model_performance": await lifecycle_manager.get_model_performance_summary(),
            "regime_analysis": await regime_detector.get_current_regime_analysis(),
            "risk_metrics": await risk_predictor.get_current_risk_metrics(),
            "feature_importance": await feature_engineer.get_feature_importance_analysis(),
            "alerts": await get_active_ml_alerts()
        }
        return dashboard_data
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_active_ml_alerts():
    """Helper function to get active ML alerts."""
    # This would integrate with your alerting system
    # For now, return a placeholder structure
    return {
        "active_alerts": [],
        "alert_count": 0,
        "last_updated": datetime.utcnow()
    }

# Configuration Endpoints
@router.get("/config")
async def get_ml_config():
    """Get current ML configuration."""
    return {
        "database_url": ml_config.database_url.replace(ml_config.database_url.split('@')[0].split('//')[1], "***"),
        "redis_url": ml_config.redis_url,
        "model_storage_path": ml_config.model_storage_path,
        "regime_detection": {
            "ensemble_size": ml_config.regime_detection.ensemble_size,
            "confidence_threshold": ml_config.regime_detection.confidence_threshold,
            "update_frequency": ml_config.regime_detection.update_frequency
        },
        "feature_engineering": {
            "cache_ttl": ml_config.feature_engineering.cache_ttl,
            "max_correlation_features": ml_config.feature_engineering.max_correlation_features
        },
        "inference": {
            "max_latency_ms": ml_config.inference.max_latency_ms,
            "cache_size": ml_config.inference.cache_size,
            "batch_size": ml_config.inference.batch_size
        }
    }

@router.post("/config/update")
async def update_ml_config(config_updates: Dict[str, Any]):
    """Update ML configuration parameters."""
    try:
        # Validate and update configuration
        updated_config = ml_config.update_config(config_updates)
        
        # Restart components with new configuration
        await restart_ml_components()
        
        return {
            "success": True,
            "updated_config": updated_config,
            "timestamp": datetime.utcnow(),
            "message": "ML configuration updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating ML config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def restart_ml_components():
    """Helper function to restart ML components with new configuration."""
    # This would restart the ML components with the new configuration
    # Implementation depends on your component lifecycle management
    pass

# Add the router to your main FastAPI app
# app.include_router(router)
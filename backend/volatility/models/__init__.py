"""
Volatility Models Module

This module contains all volatility forecasting model implementations.
"""

from .base import BaseVolatilityModel, VolatilityForecast, ModelMetrics
from .garch_models import GARCHModel, EGARCHModel, GJRGARCHModel
from .estimators import GarmanKlassEstimator, YangZhangEstimator, RogersSatchellEstimator

# Deep learning models (optional import with graceful fallback)
try:
    from .deep_learning_models import (
        LSTMVolatilityPredictor, 
        TransformerVolatilityPredictor,
        DEEP_LEARNING_AVAILABLE,
        NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
    )
    _deep_learning_exports = [
        "LSTMVolatilityPredictor",
        "TransformerVolatilityPredictor", 
        "DEEP_LEARNING_AVAILABLE",
        "NEURAL_ENGINE_OPTIMIZATION_AVAILABLE"
    ]
except ImportError:
    _deep_learning_exports = []
    DEEP_LEARNING_AVAILABLE = False
    NEURAL_ENGINE_OPTIMIZATION_AVAILABLE = False

__all__ = [
    "BaseVolatilityModel",
    "VolatilityForecast", 
    "ModelMetrics",
    "GARCHModel",
    "EGARCHModel", 
    "GJRGARCHModel",
    "GarmanKlassEstimator",
    "YangZhangEstimator",
    "RogersSatchellEstimator"
] + _deep_learning_exports
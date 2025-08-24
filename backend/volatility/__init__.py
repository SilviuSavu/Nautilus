"""
Advanced Volatility Forecasting Module for Nautilus Trading Platform

This module provides comprehensive volatility forecasting capabilities including:
- Traditional econometric models (ARCH/GARCH family)
- Stochastic volatility models (Heston, SABR)
- Deep learning approaches (LSTM, Transformer)
- Real-time volatility estimators
- Model ensemble framework
- M4 Max hardware acceleration

Author: Nautilus Trading Platform
Date: August 2025
"""

__version__ = "1.0.0"
__author__ = "Nautilus Trading Platform"

from .config import VolatilityConfig, ModelType, EstimatorType, EnsembleMethod
from .models.base import BaseVolatilityModel, VolatilityForecast, ModelMetrics

__all__ = [
    "VolatilityConfig",
    "ModelType", 
    "EstimatorType",
    "EnsembleMethod",
    "BaseVolatilityModel",
    "VolatilityForecast",
    "ModelMetrics"
]
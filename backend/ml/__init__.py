"""
Machine Learning Framework for Nautilus Trading Platform

Advanced ML enhancements including:
- Market regime detection with ensemble methods
- Real-time feature engineering and correlation analysis
- Model retraining and drift detection
- Enhanced risk prediction with ML-based portfolio optimization
- Real-time inference with <100ms latency
- Explainable AI with confidence scores
"""

__version__ = "1.0.0"
__author__ = "Nautilus ML Team"

# Core ML components
from .market_regime import MarketRegimeDetector, RegimeClassifier
from .feature_engineering import FeatureEngineer, CorrelationAnalyzer
from .model_lifecycle import ModelManager, DriftDetector, ModelRetrainer
from .risk_prediction import RiskPredictor, PortfolioOptimizer
from .inference_engine import InferenceEngine, MLMonitoringDashboard

# ML utilities
from .utils import ModelRegistry, FeatureStore, MLMetrics
from .config import MLConfig

__all__ = [
    "MarketRegimeDetector",
    "RegimeClassifier", 
    "FeatureEngineer",
    "CorrelationAnalyzer",
    "ModelManager",
    "DriftDetector",
    "ModelRetrainer",
    "RiskPredictor",
    "PortfolioOptimizer",
    "InferenceEngine",
    "MLMonitoringDashboard",
    "ModelRegistry",
    "FeatureStore",
    "MLMetrics",
    "MLConfig"
]
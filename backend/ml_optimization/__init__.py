"""
Phase 5: ML-Powered Auto-Scaling and Predictive Resource Allocation

This module implements intelligent ML-driven optimization for the Nautilus
trading platform, enhancing Phase 4 Kubernetes infrastructure with:

- Predictive auto-scaling based on trading patterns
- AI-driven performance optimization for market conditions
- Real-time resource allocation intelligence
- Market regime-aware optimization strategies

Features:
- Trading pattern prediction models
- Market volatility-based scaling
- Resource demand forecasting
- Performance optimization automation
"""

from .ml_autoscaler import MLAutoScaler
from .predictive_allocator import PredictiveResourceAllocator
from .market_optimizer import MarketConditionOptimizer
from .training_pipeline import MLTrainingPipeline
from .performance_monitor import MLPerformanceMonitor

__all__ = [
    'MLAutoScaler',
    'PredictiveResourceAllocator', 
    'MarketConditionOptimizer',
    'MLTrainingPipeline',
    'MLPerformanceMonitor'
]

__version__ = '1.0.0'
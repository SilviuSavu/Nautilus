"""
Dynamic Weight Optimization

This module implements intelligent weighting strategies for combining multiple volatility models
into ensemble forecasts with optimal performance characteristics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from ..models.base import VolatilityForecast, ModelMetrics
from ..config import EnsembleMethod, EnsembleConfig

import logging
logger = logging.getLogger(__name__)


class WeightingStrategy(Enum):
    """Available weighting strategies"""
    EQUAL = "equal"
    INVERSE_VARIANCE = "inverse_variance"
    PERFORMANCE_BASED = "performance_based"
    DYNAMIC_BAYESIAN = "dynamic_bayesian"
    REGIME_ADAPTIVE = "regime_adaptive"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class WeightingMetrics:
    """Metrics for weight optimization"""
    model_name: str
    recent_rmse: float
    recent_mae: float
    hit_ratio: float
    confidence_score: float
    stability_score: float
    recency_weight: float
    regime_consistency: float


class DynamicWeightOptimizer:
    """
    Dynamic weight optimization for model ensemble.
    
    This class implements various strategies to optimally weight different volatility models
    based on their recent performance, confidence, and market regime consistency.
    """
    
    def __init__(self, ensemble_config: EnsembleConfig):
        """
        Initialize the weight optimizer.
        
        Args:
            ensemble_config: Ensemble configuration settings
        """
        self.config = ensemble_config
        self.weight_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.regime_history: deque = deque(maxlen=20)
        
        # Weighting parameters
        self.min_weight = 0.01  # Minimum weight per model
        self.max_weight = 0.8   # Maximum weight per model
        self.decay_factor = ensemble_config.weight_decay
        self.lookback_periods = 10
        
        self.logger = logging.getLogger(__name__)
    
    async def calculate_weights(self, 
                              symbol: str,
                              forecasts: Dict[str, VolatilityForecast],
                              performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate optimal weights for model ensemble.
        
        Args:
            symbol: Trading symbol
            forecasts: Individual model forecasts
            performance_metrics: Historical performance data
            
        Returns:
            Dictionary of model weights
        """
        try:
            if not forecasts:
                return {}
            
            # Determine weighting strategy
            strategy = self._select_weighting_strategy(forecasts, performance_metrics)
            
            # Calculate weights based on strategy
            if strategy == WeightingStrategy.EQUAL:
                weights = self._calculate_equal_weights(forecasts)
            elif strategy == WeightingStrategy.INVERSE_VARIANCE:
                weights = self._calculate_inverse_variance_weights(forecasts)
            elif strategy == WeightingStrategy.PERFORMANCE_BASED:
                weights = await self._calculate_performance_weights(symbol, forecasts, performance_metrics)
            elif strategy == WeightingStrategy.DYNAMIC_BAYESIAN:
                weights = await self._calculate_bayesian_weights(symbol, forecasts, performance_metrics)
            elif strategy == WeightingStrategy.CONFIDENCE_WEIGHTED:
                weights = self._calculate_confidence_weights(forecasts)
            else:
                # Default to equal weights
                weights = self._calculate_equal_weights(forecasts)
            
            # Apply constraints and normalization
            weights = self._apply_weight_constraints(weights)
            weights = self._normalize_weights(weights)
            
            # Store weight history
            self.weight_history[symbol].append({
                'timestamp': datetime.utcnow(),
                'weights': weights.copy(),
                'strategy': strategy.value
            })
            
            self.logger.debug(f"Calculated weights for {symbol}: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating weights for {symbol}: {e}")
            # Fallback to equal weights
            return self._calculate_equal_weights(forecasts)
    
    def _select_weighting_strategy(self, 
                                  forecasts: Dict[str, VolatilityForecast],
                                  performance_metrics: Dict[str, Any]) -> WeightingStrategy:
        """
        Select optimal weighting strategy based on available data and performance.
        
        Args:
            forecasts: Model forecasts
            performance_metrics: Performance history
            
        Returns:
            Selected weighting strategy
        """
        # Use configuration default
        if self.config.method == EnsembleMethod.EQUAL_WEIGHT:
            return WeightingStrategy.EQUAL
        elif self.config.method == EnsembleMethod.VARIANCE_WEIGHT:
            return WeightingStrategy.INVERSE_VARIANCE
        elif self.config.method == EnsembleMethod.BAYESIAN_AVERAGE:
            return WeightingStrategy.DYNAMIC_BAYESIAN
        elif self.config.method == EnsembleMethod.DYNAMIC_WEIGHT:
            # Choose based on data availability
            if performance_metrics and len(performance_metrics) > 5:
                return WeightingStrategy.PERFORMANCE_BASED
            else:
                return WeightingStrategy.CONFIDENCE_WEIGHTED
        else:
            return WeightingStrategy.EQUAL
    
    def _calculate_equal_weights(self, forecasts: Dict[str, VolatilityForecast]) -> Dict[str, float]:
        """Calculate equal weights for all models"""
        n_models = len(forecasts)
        if n_models == 0:
            return {}
        
        weight = 1.0 / n_models
        return {model_name: weight for model_name in forecasts.keys()}
    
    def _calculate_inverse_variance_weights(self, forecasts: Dict[str, VolatilityForecast]) -> Dict[str, float]:
        """
        Calculate weights based on inverse variance (precision weighting).
        Models with lower forecast variance get higher weights.
        """
        weights = {}
        total_inverse_var = 0.0
        
        # Calculate inverse variances
        inverse_variances = {}
        for model_name, forecast in forecasts.items():
            # Use forecast standard error as variance proxy
            variance = max(forecast.forecast_std_error ** 2, 1e-6)  # Avoid division by zero
            inverse_var = 1.0 / variance
            inverse_variances[model_name] = inverse_var
            total_inverse_var += inverse_var
        
        # Calculate normalized weights
        if total_inverse_var > 0:
            for model_name, inverse_var in inverse_variances.items():
                weights[model_name] = inverse_var / total_inverse_var
        else:
            # Fallback to equal weights
            weights = self._calculate_equal_weights(forecasts)
        
        return weights
    
    async def _calculate_performance_weights(self, 
                                           symbol: str,
                                           forecasts: Dict[str, VolatilityForecast],
                                           performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate weights based on recent performance metrics.
        Better performing models get higher weights with recency bias.
        """
        weights = {}
        
        # Extract performance metrics
        model_metrics = {}
        for model_name in forecasts.keys():
            metrics = self._extract_model_performance(model_name, performance_metrics)
            if metrics:
                model_metrics[model_name] = metrics
        
        if not model_metrics:
            return self._calculate_equal_weights(forecasts)
        
        # Calculate performance scores
        performance_scores = {}
        for model_name, metrics in model_metrics.items():
            # Combine multiple performance measures
            rmse_score = 1.0 / (1.0 + metrics['rmse'])  # Lower RMSE is better
            hit_ratio_score = metrics.get('hit_ratio', 0.5)  # Higher hit ratio is better
            confidence_score = metrics.get('confidence', 0.5)
            
            # Weighted combination
            performance_score = (
                0.4 * rmse_score +
                0.3 * hit_ratio_score +
                0.3 * confidence_score
            )
            
            # Apply recency weighting
            recency_factor = self._calculate_recency_factor(model_name, symbol)
            performance_scores[model_name] = performance_score * recency_factor
        
        # Convert to weights
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for model_name, score in performance_scores.items():
                weights[model_name] = score / total_score
        else:
            weights = self._calculate_equal_weights(forecasts)
        
        return weights
    
    async def _calculate_bayesian_weights(self, 
                                        symbol: str,
                                        forecasts: Dict[str, VolatilityForecast],
                                        performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate Bayesian model averaging weights.
        Uses model evidence/likelihood to determine weights.
        """
        weights = {}
        
        # Extract log-likelihoods from forecasts
        log_likelihoods = {}
        for model_name, forecast in forecasts.items():
            # Use likelihood from forecast (if available)
            log_likelihood = getattr(forecast, 'likelihood', 0.0)
            log_likelihoods[model_name] = log_likelihood
        
        if not any(log_likelihoods.values()):
            # No likelihood information, fall back to performance weights
            return await self._calculate_performance_weights(symbol, forecasts, performance_metrics)
        
        # Calculate Bayesian weights using model evidence
        max_log_likelihood = max(log_likelihoods.values())
        
        # Avoid numerical overflow by subtracting max
        normalized_likelihoods = {}
        total_likelihood = 0.0
        
        for model_name, log_likelihood in log_likelihoods.items():
            likelihood = np.exp(log_likelihood - max_log_likelihood)
            normalized_likelihoods[model_name] = likelihood
            total_likelihood += likelihood
        
        # Convert to weights
        if total_likelihood > 0:
            for model_name, likelihood in normalized_likelihoods.items():
                weights[model_name] = likelihood / total_likelihood
        else:
            weights = self._calculate_equal_weights(forecasts)
        
        return weights
    
    def _calculate_confidence_weights(self, forecasts: Dict[str, VolatilityForecast]) -> Dict[str, float]:
        """
        Calculate weights based on forecast confidence intervals.
        Models with tighter confidence intervals get higher weights.
        """
        weights = {}
        
        # Calculate confidence interval widths
        ci_widths = {}
        for model_name, forecast in forecasts.items():
            ci_width = forecast.confidence_interval_upper - forecast.confidence_interval_lower
            # Use inverse of CI width (tighter intervals = higher weight)
            ci_widths[model_name] = 1.0 / max(ci_width, 1e-6)
        
        # Normalize to weights
        total_inverse_width = sum(ci_widths.values())
        if total_inverse_width > 0:
            for model_name, inverse_width in ci_widths.items():
                weights[model_name] = inverse_width / total_inverse_width
        else:
            weights = self._calculate_equal_weights(forecasts)
        
        return weights
    
    def _extract_model_performance(self, 
                                  model_name: str,
                                  performance_metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract relevant performance metrics for a model"""
        try:
            if model_name not in performance_metrics:
                return None
            
            model_perf = performance_metrics[model_name]
            
            return {
                'rmse': model_perf.get('rmse', 1.0),
                'mae': model_perf.get('mae', 1.0),
                'hit_ratio': model_perf.get('hit_ratio', 0.5),
                'confidence': model_perf.get('confidence_score', 0.5),
                'r_squared': model_perf.get('r_squared', 0.0)
            }
            
        except Exception:
            return None
    
    def _calculate_recency_factor(self, model_name: str, symbol: str) -> float:
        """
        Calculate recency weighting factor.
        More recent performance gets higher weight.
        """
        history = self.performance_history.get(f"{symbol}_{model_name}", deque())
        
        if not history:
            return 1.0  # No history, use base weight
        
        # Simple exponential decay based on time
        current_time = datetime.utcnow()
        total_weight = 0.0
        weighted_score = 0.0
        
        for i, entry in enumerate(reversed(history)):
            time_diff = (current_time - entry.get('timestamp', current_time)).total_seconds()
            days_old = time_diff / (24 * 3600)
            
            # Exponential decay
            weight = self.decay_factor ** days_old
            score = entry.get('performance_score', 1.0)
            
            weighted_score += weight * score
            total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 1.0
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum and maximum weight constraints.
        
        Args:
            weights: Raw weights
            
        Returns:
            Constrained weights
        """
        constrained_weights = {}
        
        for model_name, weight in weights.items():
            # Apply min/max constraints
            constrained_weight = max(self.min_weight, min(self.max_weight, weight))
            constrained_weights[model_name] = constrained_weight
        
        return constrained_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Weights to normalize
            
        Returns:
            Normalized weights
        """
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            # All weights are zero or negative, use equal weights
            n_models = len(weights)
            return {name: 1.0 / n_models for name in weights.keys()}
        
        # Normalize
        return {name: weight / total_weight for name, weight in weights.items()}
    
    def get_weight_history(self, symbol: str, lookback: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent weight history for a symbol.
        
        Args:
            symbol: Trading symbol
            lookback: Number of recent entries to return
            
        Returns:
            List of weight history entries
        """
        history = self.weight_history.get(symbol, deque())
        return list(history)[-lookback:]
    
    def update_performance_metrics(self, 
                                  symbol: str,
                                  model_name: str,
                                  performance_data: Dict[str, float]):
        """
        Update performance metrics for weight optimization.
        
        Args:
            symbol: Trading symbol
            model_name: Model name
            performance_data: Performance metrics
        """
        key = f"{symbol}_{model_name}"
        
        entry = {
            'timestamp': datetime.utcnow(),
            'performance_score': performance_data.get('performance_score', 1.0),
            **performance_data
        }
        
        self.performance_history[key].append(entry)
    
    def get_current_weights(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get the most recent weights for a symbol"""
        history = self.weight_history.get(symbol, deque())
        if history:
            return history[-1]['weights']
        return None
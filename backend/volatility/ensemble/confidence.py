"""
Confidence Aggregation and Uncertainty Quantification

This module implements sophisticated uncertainty quantification for ensemble volatility forecasts,
including model disagreement analysis and confidence interval aggregation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import scipy.stats as stats

from ..models.base import VolatilityForecast
from ..config import EnsembleConfig

import logging
logger = logging.getLogger(__name__)


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics for ensemble forecast"""
    ensemble_confidence: float
    forecast_uncertainty: float
    model_disagreement: float
    ci_lower: float
    ci_upper: float
    prediction_interval_width: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_level: float


class ConfidenceAggregator:
    """
    Aggregates confidence information from multiple models to provide
    comprehensive uncertainty quantification for ensemble forecasts.
    """
    
    def __init__(self, ensemble_config: EnsembleConfig):
        """
        Initialize confidence aggregator.
        
        Args:
            ensemble_config: Ensemble configuration
        """
        self.config = ensemble_config
        self.uncertainty_history: Dict[str, List[UncertaintyMetrics]] = defaultdict(list)
        self.disagreement_threshold = 0.2  # Threshold for high model disagreement
        
        self.logger = logging.getLogger(__name__)
    
    def quantify_uncertainty(self, 
                           forecasts: Dict[str, VolatilityForecast],
                           weights: Dict[str, float],
                           confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Quantify uncertainty for ensemble forecast.
        
        Args:
            forecasts: Individual model forecasts
            weights: Model weights
            confidence_level: Target confidence level
            
        Returns:
            Dictionary containing uncertainty metrics
        """
        try:
            if not forecasts or len(forecasts) < 2:
                # Not enough models for meaningful uncertainty quantification
                return self._default_uncertainty_metrics(confidence_level)
            
            # Extract forecast values and confidence intervals
            forecast_values = [f.forecast_volatility for f in forecasts.values()]
            forecast_weights = [weights.get(name, 0.0) for name in forecasts.keys()]
            
            # Calculate ensemble statistics
            weighted_mean = np.average(forecast_values, weights=forecast_weights)
            
            # Calculate different types of uncertainty
            model_disagreement = self._calculate_model_disagreement(forecasts, weights)
            epistemic_uncertainty = self._calculate_epistemic_uncertainty(forecasts, weights)
            aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(forecasts, weights)
            
            # Combine uncertainties
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            # Calculate confidence intervals
            ci_lower, ci_upper = self._calculate_ensemble_confidence_intervals(
                forecasts, weights, confidence_level
            )
            
            # Calculate overall ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(
                forecasts, weights, model_disagreement
            )
            
            # Create uncertainty metrics
            uncertainty_metrics = UncertaintyMetrics(
                ensemble_confidence=ensemble_confidence,
                forecast_uncertainty=total_uncertainty,
                model_disagreement=model_disagreement,
                ci_lower=max(0.0, ci_lower),  # Volatility can't be negative
                ci_upper=ci_upper,
                prediction_interval_width=ci_upper - ci_lower,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_level=confidence_level
            )
            
            # Convert to dictionary for API response
            return {
                'ensemble_confidence': ensemble_confidence,
                'forecast_uncertainty': total_uncertainty,
                'model_disagreement': model_disagreement,
                'ci_lower': max(0.0, ci_lower),
                'ci_upper': ci_upper,
                'prediction_interval_width': ci_upper - ci_lower,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric_uncertainty,
                'uncertainty_level': self._categorize_uncertainty_level(total_uncertainty)
            }
            
        except Exception as e:
            self.logger.error(f"Error quantifying uncertainty: {e}")
            return self._default_uncertainty_metrics(confidence_level)
    
    def _calculate_model_disagreement(self, 
                                    forecasts: Dict[str, VolatilityForecast],
                                    weights: Dict[str, float]) -> float:
        """
        Calculate model disagreement as weighted variance of forecasts.
        
        Args:
            forecasts: Individual model forecasts
            weights: Model weights
            
        Returns:
            Model disagreement metric (higher = more disagreement)
        """
        forecast_values = np.array([f.forecast_volatility for f in forecasts.values()])
        forecast_weights = np.array([weights.get(name, 0.0) for name in forecasts.keys()])
        
        # Weighted mean
        weighted_mean = np.average(forecast_values, weights=forecast_weights)
        
        # Weighted variance (measure of disagreement)
        if len(forecast_values) > 1:
            weighted_variance = np.average((forecast_values - weighted_mean)**2, weights=forecast_weights)
            return np.sqrt(weighted_variance)  # Standard deviation
        else:
            return 0.0
    
    def _calculate_epistemic_uncertainty(self, 
                                       forecasts: Dict[str, VolatilityForecast],
                                       weights: Dict[str, float]) -> float:
        """
        Calculate epistemic (model) uncertainty - uncertainty due to model choice.
        
        This represents the uncertainty arising from not knowing which model is correct.
        """
        # Model disagreement is a good proxy for epistemic uncertainty
        return self._calculate_model_disagreement(forecasts, weights)
    
    def _calculate_aleatoric_uncertainty(self, 
                                       forecasts: Dict[str, VolatilityForecast],
                                       weights: Dict[str, float]) -> float:
        """
        Calculate aleatoric (data) uncertainty - irreducible uncertainty in the data.
        
        This represents the uncertainty that remains even with perfect model knowledge.
        """
        # Use weighted average of individual model uncertainties
        uncertainties = []
        forecast_weights = []
        
        for name, forecast in forecasts.items():
            weight = weights.get(name, 0.0)
            if weight > 0:
                # Use forecast standard error as uncertainty measure
                uncertainty = forecast.forecast_std_error
                uncertainties.append(uncertainty)
                forecast_weights.append(weight)
        
        if uncertainties:
            return np.average(uncertainties, weights=forecast_weights)
        else:
            return 0.01  # Default uncertainty
    
    def _calculate_ensemble_confidence_intervals(self, 
                                               forecasts: Dict[str, VolatilityForecast],
                                               weights: Dict[str, float],
                                               confidence_level: float) -> Tuple[float, float]:
        """
        Calculate confidence intervals for ensemble forecast.
        
        Combines individual model confidence intervals using weighted approach.
        """
        # Approach 1: Weighted combination of individual CIs
        weighted_lower = 0.0
        weighted_upper = 0.0
        total_weight = 0.0
        
        for name, forecast in forecasts.items():
            weight = weights.get(name, 0.0)
            if weight > 0:
                weighted_lower += weight * forecast.confidence_interval_lower
                weighted_upper += weight * forecast.confidence_interval_upper
                total_weight += weight
        
        if total_weight > 0:
            ci_lower_v1 = weighted_lower / total_weight
            ci_upper_v1 = weighted_upper / total_weight
        else:
            ci_lower_v1 = ci_upper_v1 = 0.0
        
        # Approach 2: Use ensemble uncertainty
        forecast_values = [f.forecast_volatility for f in forecasts.values()]
        forecast_weights = [weights.get(name, 0.0) for name in forecasts.keys()]
        ensemble_mean = np.average(forecast_values, weights=forecast_weights)
        
        # Calculate ensemble standard error
        model_disagreement = self._calculate_model_disagreement(forecasts, weights)
        avg_individual_error = np.average([f.forecast_std_error for f in forecasts.values()], 
                                        weights=forecast_weights)
        
        # Combine model disagreement and individual uncertainties
        ensemble_std_error = np.sqrt(model_disagreement**2 + avg_individual_error**2)
        
        # Calculate confidence intervals using normal approximation
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        ci_lower_v2 = ensemble_mean - z_score * ensemble_std_error
        ci_upper_v2 = ensemble_mean + z_score * ensemble_std_error
        
        # Use the more conservative (wider) intervals
        ci_lower = min(ci_lower_v1, ci_lower_v2)
        ci_upper = max(ci_upper_v1, ci_upper_v2)
        
        return ci_lower, ci_upper
    
    def _calculate_ensemble_confidence(self, 
                                     forecasts: Dict[str, VolatilityForecast],
                                     weights: Dict[str, float],
                                     model_disagreement: float) -> float:
        """
        Calculate overall ensemble confidence score.
        
        Higher confidence when:
        - Models agree (low disagreement)
        - Individual models are confident
        - Weights are well-distributed
        """
        # Component 1: Model agreement (inverse of disagreement)
        max_disagreement = 0.5  # Normalize disagreement to 0-1 scale
        agreement_score = max(0.0, 1.0 - (model_disagreement / max_disagreement))
        
        # Component 2: Average individual model confidence
        # (Use inverse of standard errors as confidence proxy)
        individual_confidences = []
        forecast_weights = []
        
        for name, forecast in forecasts.items():
            weight = weights.get(name, 0.0)
            if weight > 0:
                # Higher confidence = lower standard error
                confidence = 1.0 / (1.0 + forecast.forecast_std_error)
                individual_confidences.append(confidence)
                forecast_weights.append(weight)
        
        if individual_confidences:
            avg_confidence = np.average(individual_confidences, weights=forecast_weights)
        else:
            avg_confidence = 0.5
        
        # Component 3: Weight distribution (more uniform = more confident)
        if len(weights) > 1:
            weight_values = list(weights.values())
            weight_entropy = -sum(w * np.log(w + 1e-10) for w in weight_values if w > 0)
            max_entropy = np.log(len(weight_values))
            weight_distribution_score = weight_entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            weight_distribution_score = 0.0
        
        # Combine components
        ensemble_confidence = (
            0.5 * agreement_score +
            0.4 * avg_confidence +
            0.1 * weight_distribution_score
        )
        
        return max(0.0, min(1.0, ensemble_confidence))  # Clip to [0, 1]
    
    def _categorize_uncertainty_level(self, total_uncertainty: float) -> str:
        """
        Categorize uncertainty level for interpretability.
        
        Args:
            total_uncertainty: Total uncertainty value
            
        Returns:
            Uncertainty level category
        """
        if total_uncertainty < 0.05:
            return "LOW"
        elif total_uncertainty < 0.15:
            return "MEDIUM"
        elif total_uncertainty < 0.30:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _default_uncertainty_metrics(self, confidence_level: float) -> Dict[str, float]:
        """Return default uncertainty metrics when calculation fails"""
        return {
            'ensemble_confidence': 0.5,
            'forecast_uncertainty': 0.1,
            'model_disagreement': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 1.0,
            'prediction_interval_width': 1.0,
            'epistemic_uncertainty': 0.05,
            'aleatoric_uncertainty': 0.05,
            'uncertainty_level': 'MEDIUM'
        }
    
    def analyze_model_reliability(self, 
                                 forecasts: Dict[str, VolatilityForecast],
                                 weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze individual model reliability for the current forecast.
        
        Args:
            forecasts: Individual model forecasts
            weights: Model weights
            
        Returns:
            Model reliability analysis
        """
        reliability_analysis = {}
        
        for name, forecast in forecasts.items():
            weight = weights.get(name, 0.0)
            
            # Calculate reliability metrics
            ci_width = forecast.confidence_interval_upper - forecast.confidence_interval_lower
            relative_uncertainty = forecast.forecast_std_error / max(forecast.forecast_volatility, 1e-6)
            
            reliability_score = min(1.0, weight * 2.0)  # Weight as proxy for reliability
            
            reliability_analysis[name] = {
                'weight': weight,
                'forecast_value': forecast.forecast_volatility,
                'standard_error': forecast.forecast_std_error,
                'confidence_interval_width': ci_width,
                'relative_uncertainty': relative_uncertainty,
                'reliability_score': reliability_score,
                'model_confidence': 1.0 / (1.0 + relative_uncertainty)
            }
        
        return reliability_analysis
    
    def detect_forecast_anomalies(self, 
                                 forecasts: Dict[str, VolatilityForecast],
                                 symbol: str) -> List[Dict[str, Any]]:
        """
        Detect potential anomalies in model forecasts.
        
        Args:
            forecasts: Individual model forecasts
            symbol: Trading symbol
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(forecasts) < 3:
            return anomalies  # Need at least 3 models for anomaly detection
        
        forecast_values = np.array([f.forecast_volatility for f in forecasts.values()])
        model_names = list(forecasts.keys())
        
        # Statistical outlier detection using IQR method
        q1 = np.percentile(forecast_values, 25)
        q3 = np.percentile(forecast_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i, (name, forecast) in enumerate(forecasts.items()):
            value = forecast.forecast_volatility
            
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    'model_name': name,
                    'forecast_value': value,
                    'deviation_type': 'outlier',
                    'severity': 'high' if abs(value - np.median(forecast_values)) > 2 * iqr else 'medium',
                    'description': f'{name} forecast ({value:.4f}) is an outlier'
                })
        
        # Check for extremely wide confidence intervals
        for name, forecast in forecasts.items():
            ci_width = forecast.confidence_interval_upper - forecast.confidence_interval_lower
            median_ci_width = np.median([
                f.confidence_interval_upper - f.confidence_interval_lower 
                for f in forecasts.values()
            ])
            
            if ci_width > 3 * median_ci_width:
                anomalies.append({
                    'model_name': name,
                    'forecast_value': forecast.forecast_volatility,
                    'deviation_type': 'high_uncertainty',
                    'severity': 'medium',
                    'description': f'{name} has unusually wide confidence interval ({ci_width:.4f})'
                })
        
        return anomalies
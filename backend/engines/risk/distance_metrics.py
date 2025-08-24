#!/usr/bin/env python3
"""
Distance Metrics for Supervised k-NN Portfolio Optimization
Implements Hassanat distance and other specialized distance measures for financial data
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Supported distance metrics for k-NN portfolio optimization"""
    EUCLIDEAN = "euclidean"
    HASSANAT = "hassanat"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    MINKOWSKI = "minkowski"
    MAHALANOBIS = "mahalanobis"
    CORRELATION = "correlation"
    FINANCIAL_WEIGHTED = "financial_weighted"

@dataclass
class DistanceConfig:
    """Configuration for distance metric calculations"""
    metric: DistanceMetric = DistanceMetric.HASSANAT
    normalize_features: bool = True
    feature_weights: Optional[Dict[str, float]] = None
    minkowski_p: float = 2.0
    correlation_threshold: float = 0.1
    handle_missing_values: bool = True
    missing_value_strategy: str = "mean"  # 'mean', 'median', 'zero'

class DistanceCalculator:
    """
    Advanced distance calculator for financial features in supervised portfolio optimization
    
    Implements multiple distance metrics optimized for financial time series data:
    - Hassanat distance: Scale-invariant, ideal for financial ratios
    - Financial weighted distance: Custom weights for financial importance
    - Traditional metrics: Euclidean, Manhattan, etc.
    """
    
    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()
        self._feature_cache = {}
        self._normalization_params = {}
        
    def compute_distance(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Compute distance between two feature vectors using configured metric
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Distance value (lower = more similar)
        """
        try:
            # Convert to aligned arrays
            array1, array2, feature_names = self._align_feature_vectors(features1, features2)
            
            # Handle missing values
            if self.config.handle_missing_values:
                array1, array2 = self._handle_missing_values(array1, array2)
            
            # Normalize features if requested
            if self.config.normalize_features:
                array1, array2 = self._normalize_features(array1, array2, feature_names)
            
            # Apply feature weights if specified
            if self.config.feature_weights:
                array1, array2 = self._apply_feature_weights(array1, array2, feature_names)
            
            # Compute distance using specified metric
            if self.config.metric == DistanceMetric.HASSANAT:
                return self._hassanat_distance(array1, array2)
            elif self.config.metric == DistanceMetric.EUCLIDEAN:
                return self._euclidean_distance(array1, array2)
            elif self.config.metric == DistanceMetric.MANHATTAN:
                return self._manhattan_distance(array1, array2)
            elif self.config.metric == DistanceMetric.COSINE:
                return self._cosine_distance(array1, array2)
            elif self.config.metric == DistanceMetric.MINKOWSKI:
                return self._minkowski_distance(array1, array2, self.config.minkowski_p)
            elif self.config.metric == DistanceMetric.CORRELATION:
                return self._correlation_distance(array1, array2)
            elif self.config.metric == DistanceMetric.FINANCIAL_WEIGHTED:
                return self._financial_weighted_distance(array1, array2, feature_names)
            else:
                raise ValueError(f"Unsupported distance metric: {self.config.metric}")
                
        except Exception as e:
            logger.error(f"Distance computation error: {e}")
            return float('inf')  # Return maximum distance on error
    
    def _hassanat_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """
        Compute Hassanat distance - scale-invariant metric ideal for financial data
        
        Formula: d(x,y) = Σ min(xi,yi) / max(xi,yi) for xi,yi > 0
                       = Σ |xi - yi| for xi,yi ≤ 0 or if max(xi,yi) = 0
        
        Reference: "A novel similarity measure for improved classification" (2016)
        """
        distance = 0.0
        n_features = len(array1)
        
        for i in range(n_features):
            x, y = array1[i], array2[i]
            
            # Handle different cases as per Hassanat distance definition
            if x > 0 and y > 0:
                # Both positive: use ratio-based distance
                min_val = min(x, y)
                max_val = max(x, y)
                distance += 1 - (min_val / max_val)
            elif x == 0 and y == 0:
                # Both zero: perfect similarity
                distance += 0
            else:
                # Handle negative or mixed cases with absolute difference
                max_abs = max(abs(x), abs(y))
                if max_abs > 0:
                    distance += abs(x - y) / max_abs
                else:
                    distance += abs(x - y)
        
        return distance
    
    def _euclidean_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Compute Euclidean distance"""
        return np.sqrt(np.sum((array1 - array2) ** 2))
    
    def _manhattan_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Compute Manhattan (L1) distance"""
        return np.sum(np.abs(array1 - array2))
    
    def _cosine_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Compute cosine distance (1 - cosine similarity)"""
        dot_product = np.dot(array1, array2)
        norm_product = np.linalg.norm(array1) * np.linalg.norm(array2)
        
        if norm_product == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_similarity = dot_product / norm_product
        return 1 - cosine_similarity
    
    def _minkowski_distance(self, array1: np.ndarray, array2: np.ndarray, p: float) -> float:
        """Compute Minkowski distance with parameter p"""
        return np.sum(np.abs(array1 - array2) ** p) ** (1/p)
    
    def _correlation_distance(self, array1: np.ndarray, array2: np.ndarray) -> float:
        """Compute correlation-based distance"""
        if len(array1) < 2:
            return self._euclidean_distance(array1, array2)
        
        correlation = np.corrcoef(array1, array2)[0, 1]
        
        # Handle NaN correlation (constant arrays)
        if np.isnan(correlation):
            return self._euclidean_distance(array1, array2)
        
        # Convert correlation to distance (0 = perfect correlation, 2 = perfect anti-correlation)
        return 1 - correlation
    
    def _financial_weighted_distance(self, array1: np.ndarray, array2: np.ndarray, 
                                   feature_names: List[str]) -> float:
        """
        Compute financial-specific weighted distance with emphasis on key risk metrics
        
        Gives higher weights to:
        - Volatility measures
        - Correlation structure
        - Return characteristics
        - Risk-adjusted metrics
        """
        # Default financial weights
        financial_weights = {
            'returns_volatility': 2.0,
            'returns_skewness': 1.5,
            'returns_kurtosis': 1.5,
            'correlation': 2.0,
            'sharpe_ratio': 2.5,
            'max_drawdown': 2.5,
            'var_95': 2.0,
            'momentum': 1.2,
            'volatility_regime': 1.8,
            'default': 1.0
        }
        
        total_distance = 0.0
        total_weight = 0.0
        
        for i, feature_name in enumerate(feature_names):
            # Determine weight for this feature
            weight = 1.0
            for key, w in financial_weights.items():
                if key.lower() in feature_name.lower():
                    weight = w
                    break
            else:
                weight = financial_weights['default']
            
            # Compute weighted Hassanat distance for this feature
            x, y = array1[i], array2[i]
            feature_distance = self._single_feature_hassanat_distance(x, y)
            
            total_distance += weight * feature_distance
            total_weight += weight
        
        return total_distance / total_weight if total_weight > 0 else 0.0
    
    def _single_feature_hassanat_distance(self, x: float, y: float) -> float:
        """Compute Hassanat distance for a single feature pair"""
        if x > 0 and y > 0:
            return 1 - (min(x, y) / max(x, y))
        elif x == 0 and y == 0:
            return 0.0
        else:
            max_abs = max(abs(x), abs(y))
            if max_abs > 0:
                return abs(x - y) / max_abs
            else:
                return abs(x - y)
    
    def _align_feature_vectors(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Align two feature dictionaries into numpy arrays with consistent ordering
        
        Returns:
            Tuple of (array1, array2, feature_names)
        """
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            raise ValueError("No common features found between vectors")
        
        # Sort for consistent ordering
        feature_names = sorted(common_features)
        
        # Extract values
        array1 = np.array([self._extract_numeric_value(features1[name]) for name in feature_names])
        array2 = np.array([self._extract_numeric_value(features2[name]) for name in feature_names])
        
        return array1, array2, feature_names
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from various data types"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) > 0:
                return float(value[0]) if np.isscalar(value[0]) else float(np.mean(value))
            else:
                return 0.0
        elif isinstance(value, dict):
            # For nested dictionaries, take the first numeric value
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {value} to float, using 0.0")
                return 0.0
    
    def _handle_missing_values(self, array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values in feature arrays"""
        # Find indices with missing values (NaN or inf)
        mask1 = np.isfinite(array1)
        mask2 = np.isfinite(array2)
        valid_mask = mask1 & mask2
        
        if not valid_mask.any():
            logger.warning("All features have missing values, using zeros")
            return np.zeros_like(array1), np.zeros_like(array2)
        
        # Replace missing values based on strategy
        if self.config.missing_value_strategy == "mean":
            fill_value = np.mean(np.concatenate([array1[mask1], array2[mask2]]))
        elif self.config.missing_value_strategy == "median":
            fill_value = np.median(np.concatenate([array1[mask1], array2[mask2]]))
        else:  # zero
            fill_value = 0.0
        
        array1_filled = np.where(mask1, array1, fill_value)
        array2_filled = np.where(mask2, array2, fill_value)
        
        return array1_filled, array2_filled
    
    def _normalize_features(self, array1: np.ndarray, array2: np.ndarray, 
                          feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features to have similar scales
        Uses cached normalization parameters for consistency
        """
        cache_key = tuple(feature_names)
        
        if cache_key not in self._normalization_params:
            # Compute normalization parameters from both arrays
            combined = np.vstack([array1, array2])
            mean_vals = np.mean(combined, axis=0)
            std_vals = np.std(combined, axis=0)
            
            # Prevent division by zero
            std_vals = np.where(std_vals == 0, 1.0, std_vals)
            
            self._normalization_params[cache_key] = (mean_vals, std_vals)
        
        mean_vals, std_vals = self._normalization_params[cache_key]
        
        # Normalize both arrays
        array1_norm = (array1 - mean_vals) / std_vals
        array2_norm = (array2 - mean_vals) / std_vals
        
        return array1_norm, array2_norm
    
    def _apply_feature_weights(self, array1: np.ndarray, array2: np.ndarray, 
                             feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature weights to arrays"""
        weights = np.ones(len(feature_names))
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.config.feature_weights:
                weights[i] = self.config.feature_weights[feature_name]
        
        # Apply square root of weights to preserve distance properties
        weight_factors = np.sqrt(weights)
        
        return array1 * weight_factors, array2 * weight_factors
    
    def compute_distance_matrix(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute pairwise distance matrix for a list of feature vectors
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Symmetric distance matrix
        """
        n = len(features_list)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.compute_distance(features_list[i], features_list[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def find_k_nearest_neighbors(self, query_features: Dict[str, Any], 
                               training_features: List[Dict[str, Any]], 
                               k: int) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors for query features
        
        Returns:
            List of (index, distance) tuples sorted by distance
        """
        distances = []
        
        for i, train_features in enumerate(training_features):
            dist = self.compute_distance(query_features, train_features)
            distances.append((i, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_distance_statistics(self, features_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute distance statistics for a feature set
        
        Useful for understanding the distribution of similarities in the training data
        """
        if len(features_list) < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        
        distances = []
        n = len(features_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.compute_distance(features_list[i], features_list[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "median": float(np.median(distances)),
            "q25": float(np.percentile(distances, 25)),
            "q75": float(np.percentile(distances, 75)),
            "count": len(distances)
        }

def create_distance_calculator(metric: str = "hassanat", 
                             normalize: bool = True,
                             feature_weights: Dict[str, float] = None) -> DistanceCalculator:
    """
    Factory function to create a distance calculator with common configurations
    
    Args:
        metric: Distance metric name
        normalize: Whether to normalize features
        feature_weights: Optional feature weights
    
    Returns:
        Configured DistanceCalculator instance
    """
    config = DistanceConfig(
        metric=DistanceMetric(metric),
        normalize_features=normalize,
        feature_weights=feature_weights
    )
    
    return DistanceCalculator(config)

# Pre-configured calculators for common use cases
def get_default_financial_calculator() -> DistanceCalculator:
    """Get calculator optimized for financial features"""
    return create_distance_calculator("hassanat", True)

def get_correlation_calculator() -> DistanceCalculator:
    """Get calculator using correlation-based distance"""
    return create_distance_calculator("correlation", True)

def get_robust_calculator() -> DistanceCalculator:
    """Get calculator robust to outliers and missing values"""
    config = DistanceConfig(
        metric=DistanceMetric.HASSANAT,
        normalize_features=True,
        handle_missing_values=True,
        missing_value_strategy="median"
    )
    return DistanceCalculator(config)
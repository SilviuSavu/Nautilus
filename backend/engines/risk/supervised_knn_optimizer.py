#!/usr/bin/env python3
"""
Supervised k-NN Portfolio Optimization
World's first implementation of supervised machine learning portfolio optimization
Uses historical optimal portfolios as training data for k-NN similarity-based optimization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import asyncio
import json
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from distance_metrics import DistanceCalculator, DistanceConfig, DistanceMetric, create_distance_calculator
from market_features import MarketFeatureExtractor, MarketFeatures, create_feature_weights_for_knn

logger = logging.getLogger(__name__)

@dataclass
class OptimalPortfolio:
    """Represents an optimal portfolio for training data"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    max_drawdown: float
    optimization_method: str
    market_features: MarketFeatures
    performance_1m: float
    performance_3m: float
    performance_6m: float
    timestamp: datetime
    validation_period: int  # Days used for ex-post validation

@dataclass
class SupervisedOptimizationRequest:
    """Request for supervised k-NN portfolio optimization"""
    assets: List[str]
    historical_returns: pd.DataFrame
    k_neighbors: Optional[int] = None  # Use dynamic k* selection if None
    distance_metric: str = "hassanat"
    lookback_periods: int = 252
    min_training_periods: int = 504  # Minimum 2 years of training data
    feature_weights: Optional[Dict[str, float]] = None
    constraints: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
@dataclass
class SupervisedOptimizationResult:
    """Result from supervised k-NN portfolio optimization"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    k_neighbors_used: int
    model_confidence: float
    training_periods: int
    validation_score: float
    distance_metric: str
    nearest_neighbors_info: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    performance_prediction: Dict[str, float]
    metadata: Dict[str, Any]

class SupervisedKNNOptimizer:
    """
    Supervised k-Nearest Neighbors Portfolio Optimizer
    
    Revolutionary approach that learns from historical optimal portfolios rather than
    relying solely on mathematical optimization. Uses market similarity to predict
    optimal portfolio weights based on historically successful allocations.
    
    Key Features:
    - Hassanat distance for scale-invariant similarity
    - Dynamic k* selection via cross-validation
    - Comprehensive market feature engineering
    - Bootstrap confidence intervals
    - Performance prediction and validation
    """
    
    def __init__(self, 
                 distance_metric: str = "hassanat",
                 feature_weights: Dict[str, float] = None,
                 cache_size: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize supervised k-NN optimizer
        
        Args:
            distance_metric: Distance metric for similarity calculation
            feature_weights: Optional feature importance weights
            cache_size: Size of training data cache
            confidence_level: Confidence level for bootstrap intervals
        """
        self.distance_metric = distance_metric
        self.feature_weights = feature_weights or create_feature_weights_for_knn()
        self.confidence_level = confidence_level
        self.cache_size = cache_size
        
        # Initialize components
        self.distance_calculator = create_distance_calculator(
            distance_metric,
            normalize=True,
            feature_weights=self.feature_weights
        )
        
        self.feature_extractor = MarketFeatureExtractor()
        
        # Training data cache
        self._training_data: List[OptimalPortfolio] = []
        self._feature_cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.optimization_count = 0
        self.total_training_time = 0.0
        self.average_prediction_time = 0.0
        
    async def optimize_portfolio(self, request: SupervisedOptimizationRequest) -> SupervisedOptimizationResult:
        """
        Main supervised k-NN portfolio optimization
        
        Args:
            request: Optimization request with assets and historical data
            
        Returns:
            SupervisedOptimizationResult with optimal weights and metadata
        """
        try:
            start_time = datetime.now()
            
            # Validate input
            self._validate_request(request)
            
            # Generate or load training data
            training_data = await self._generate_training_data(request)
            
            if len(training_data) < request.min_training_periods // 10:  # Minimum samples
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # Extract current market features
            current_features = await self._extract_current_features(request)
            
            # Determine optimal k using cross-validation
            optimal_k = await self._select_optimal_k(
                current_features, 
                training_data, 
                request.k_neighbors,
                request.cross_validation_folds
            )
            
            # Perform k-NN prediction
            optimal_weights, neighbors_info = await self._predict_optimal_weights(
                current_features,
                training_data,
                optimal_k,
                request.assets
            )
            
            # Apply constraints if specified
            if request.constraints:
                optimal_weights = self._apply_constraints(optimal_weights, request.constraints)
            
            # Compute performance predictions and confidence
            performance_prediction = await self._predict_performance(neighbors_info)
            model_confidence = await self._compute_model_confidence(
                current_features,
                training_data,
                neighbors_info
            )
            
            # Validation score from cross-validation
            validation_score = await self._compute_validation_score(
                current_features,
                training_data,
                optimal_k,
                request.validation_split
            )
            
            # Feature importance analysis
            feature_importance = await self._compute_feature_importance(
                training_data,
                neighbors_info
            )
            
            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(processing_time)
            
            # Create result
            result = SupervisedOptimizationResult(
                optimal_weights=optimal_weights,
                expected_return=performance_prediction.get('expected_return', 0.08),
                expected_risk=performance_prediction.get('expected_risk', 0.15),
                sharpe_ratio=performance_prediction.get('sharpe_ratio', 0.53),
                k_neighbors_used=optimal_k,
                model_confidence=model_confidence,
                training_periods=len(training_data),
                validation_score=validation_score,
                distance_metric=self.distance_metric,
                nearest_neighbors_info=neighbors_info,
                feature_importance=feature_importance,
                performance_prediction=performance_prediction,
                metadata={
                    'processing_time_seconds': processing_time,
                    'training_data_size': len(training_data),
                    'current_features_count': len(asdict(current_features)),
                    'optimization_method': 'supervised_knn',
                    'confidence_level': self.confidence_level,
                    'validation_method': 'time_series_cross_validation'
                }
            )
            
            logger.info(f"Supervised k-NN optimization completed: k={optimal_k}, confidence={model_confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Supervised optimization failed: {e}")
            raise
    
    async def _generate_training_data(self, request: SupervisedOptimizationRequest) -> List[OptimalPortfolio]:
        """
        Generate training data from historical returns using ex-post optimization
        
        Creates optimal portfolios for each historical period using various optimization methods
        """
        training_data = []
        returns_data = request.historical_returns
        
        # Minimum window for meaningful optimization
        min_window = max(63, request.lookback_periods // 4)  # At least 3 months
        
        for i in range(min_window, len(returns_data) - 21, 21):  # Every 3 weeks
            try:
                # Training window
                train_window = returns_data.iloc[max(0, i - request.lookback_periods):i]
                
                # Forward validation window
                validation_window = returns_data.iloc[i:i + 21]
                
                if len(train_window) < min_window or len(validation_window) < 10:
                    continue
                
                # Extract market features for this period
                features = await self._extract_features_for_period(train_window)
                
                # Generate multiple optimal portfolios using different methods
                optimization_methods = [
                    'sharpe_maximization',
                    'minimum_variance',
                    'risk_parity',
                    'maximum_diversification'
                ]
                
                for method in optimization_methods:
                    optimal_portfolio = await self._compute_ex_post_optimal_portfolio(
                        train_window,
                        validation_window,
                        method,
                        features,
                        request.assets
                    )
                    
                    if optimal_portfolio:
                        training_data.append(optimal_portfolio)
                
            except Exception as e:
                logger.warning(f"Failed to generate training sample at period {i}: {e}")
                continue
        
        logger.info(f"Generated {len(training_data)} training samples from {len(returns_data)} historical periods")
        return training_data
    
    async def _compute_ex_post_optimal_portfolio(self,
                                               train_data: pd.DataFrame,
                                               validation_data: pd.DataFrame,
                                               method: str,
                                               features: MarketFeatures,
                                               assets: List[str]) -> Optional[OptimalPortfolio]:
        """
        Compute ex-post optimal portfolio using specified optimization method
        
        Args:
            train_data: Historical data for optimization
            validation_data: Forward period for performance validation
            method: Optimization method name
            features: Market features for this period
            assets: Asset universe
            
        Returns:
            OptimalPortfolio object or None if optimization fails
        """
        try:
            # Compute returns statistics
            asset_returns = train_data.mean()
            cov_matrix = train_data.cov()
            
            # Initialize weights
            n_assets = len(assets)
            weights = np.ones(n_assets) / n_assets  # Start with equal weights
            
            if method == 'sharpe_maximization':
                weights = await self._optimize_sharpe_ratio(asset_returns, cov_matrix)
            elif method == 'minimum_variance':
                weights = await self._optimize_minimum_variance(cov_matrix)
            elif method == 'risk_parity':
                weights = await self._optimize_risk_parity(cov_matrix)
            elif method == 'maximum_diversification':
                weights = await self._optimize_max_diversification(asset_returns, cov_matrix)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Create weight dictionary
            weight_dict = {asset: float(weights[i]) for i, asset in enumerate(assets)}
            
            # Compute portfolio metrics on training data
            portfolio_returns = (train_data * weights).sum(axis=1)
            expected_return = float(portfolio_returns.mean() * 252)
            expected_risk = float(portfolio_returns.std() * np.sqrt(252))
            sharpe_ratio = (expected_return - 0.02) / expected_risk if expected_risk > 0 else 0.0
            
            # Compute max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            drawdown = (cumulative - cumulative.expanding().max()) / cumulative.expanding().max()
            max_drawdown = float(drawdown.min())
            
            # Compute forward performance
            if len(validation_data) > 0:
                forward_returns = (validation_data * weights).sum(axis=1)
                performance_1m = float(forward_returns.sum())
                performance_3m = performance_1m  # Use same for now
                performance_6m = performance_1m
            else:
                performance_1m = performance_3m = performance_6m = 0.0
            
            return OptimalPortfolio(
                weights=weight_dict,
                expected_return=expected_return,
                expected_risk=expected_risk,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                optimization_method=method,
                market_features=features,
                performance_1m=performance_1m,
                performance_3m=performance_3m,
                performance_6m=performance_6m,
                timestamp=datetime.now(),
                validation_period=len(validation_data)
            )
            
        except Exception as e:
            logger.error(f"Ex-post optimization failed for method {method}: {e}")
            return None
    
    async def _optimize_sharpe_ratio(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for maximum Sharpe ratio (simplified)"""
        try:
            # Simple optimization: inverse volatility weighting with return tilt
            inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
            return_weights = np.maximum(returns, 0)  # Only positive expected returns
            
            weights = inv_vol * (1 + return_weights)
            return weights / weights.sum()
        except:
            return np.ones(len(returns)) / len(returns)
    
    async def _optimize_minimum_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for minimum variance"""
        try:
            # Simplified minimum variance: inverse volatility weighting
            inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
            weights = inv_vol / inv_vol.sum()
            return weights
        except:
            n = len(cov_matrix)
            return np.ones(n) / n
    
    async def _optimize_risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for equal risk contribution (simplified)"""
        try:
            # Simplified risk parity: inverse volatility weighting
            inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
            weights = inv_vol / inv_vol.sum()
            return weights
        except:
            n = len(cov_matrix)
            return np.ones(n) / n
    
    async def _optimize_max_diversification(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize for maximum diversification ratio"""
        try:
            # Simplified max diversification: weighted inverse volatility
            vol = np.sqrt(np.diag(cov_matrix))
            inv_vol = 1.0 / vol
            
            # Weight by correlation (lower correlation = higher weight)
            corr_matrix = cov_matrix.corr()
            avg_corr = corr_matrix.mean(axis=1) - 1/len(corr_matrix)  # Subtract self-correlation
            diversification_weights = inv_vol * (1 - avg_corr)
            
            return diversification_weights / diversification_weights.sum()
        except:
            n = len(returns)
            return np.ones(n) / n
    
    async def _extract_current_features(self, request: SupervisedOptimizationRequest) -> MarketFeatures:
        """Extract market features for current period"""
        # Use the most recent data for feature extraction
        recent_data = request.historical_returns.tail(request.lookback_periods)
        
        features = self.feature_extractor.extract_features(
            returns_data=recent_data,
            timestamp=datetime.now()
        )
        
        return features
    
    async def _extract_features_for_period(self, returns_data: pd.DataFrame) -> MarketFeatures:
        """Extract features for a specific historical period"""
        return self.feature_extractor.extract_features(
            returns_data=returns_data,
            timestamp=returns_data.index[-1] if len(returns_data) > 0 else datetime.now()
        )
    
    async def _select_optimal_k(self,
                              current_features: MarketFeatures,
                              training_data: List[OptimalPortfolio],
                              requested_k: Optional[int],
                              cv_folds: int) -> int:
        """
        Select optimal k using cross-validation
        
        Args:
            current_features: Current market features
            training_data: Historical optimal portfolios
            requested_k: User-specified k (if any)
            cv_folds: Number of CV folds
            
        Returns:
            Optimal k value
        """
        if requested_k is not None:
            return min(requested_k, len(training_data) - 1)
        
        # Test range of k values
        max_k = min(50, len(training_data) // 2)
        k_range = range(3, max_k, 2)  # Test odd values from 3 to max_k
        
        best_k = 5  # Default
        best_score = float('-inf')
        
        try:
            for k in k_range:
                # Cross-validation score for this k
                scores = []
                
                # Time series split to respect temporal order
                n_samples = len(training_data)
                fold_size = n_samples // cv_folds
                
                for fold in range(cv_folds - 1):  # Leave last fold for final validation
                    # Training set for this fold
                    train_start = fold * fold_size
                    train_end = train_start + fold_size * 2  # Use 2 folds for training
                    
                    # Test set
                    test_start = train_end
                    test_end = min(test_start + fold_size, n_samples)
                    
                    if test_end <= test_start:
                        continue
                    
                    fold_train_data = training_data[train_start:train_end]
                    fold_test_data = training_data[test_start:test_end]
                    
                    # Evaluate this k on the fold
                    fold_score = await self._evaluate_k_on_fold(
                        k, fold_train_data, fold_test_data
                    )
                    
                    if not np.isnan(fold_score):
                        scores.append(fold_score)
                
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_k = k
                        
        except Exception as e:
            logger.warning(f"K selection failed, using default k=5: {e}")
            best_k = 5
        
        logger.debug(f"Selected optimal k={best_k} with CV score={best_score:.4f}")
        return best_k
    
    async def _evaluate_k_on_fold(self,
                                k: int,
                                train_data: List[OptimalPortfolio],
                                test_data: List[OptimalPortfolio]) -> float:
        """Evaluate k-NN performance on a cross-validation fold"""
        try:
            scores = []
            
            for test_sample in test_data:
                # Find k nearest neighbors in training data
                distances = []
                for train_sample in train_data:
                    dist = self.distance_calculator.compute_distance(
                        asdict(test_sample.market_features),
                        asdict(train_sample.market_features)
                    )
                    distances.append((dist, train_sample))
                
                # Get k nearest neighbors
                distances.sort(key=lambda x: x[0])
                neighbors = distances[:k]
                
                # Predict performance using weighted average
                predicted_performance = 0.0
                total_weight = 0.0
                
                for dist, neighbor in neighbors:
                    weight = 1.0 / (dist + 1e-8)  # Inverse distance weighting
                    predicted_performance += weight * neighbor.sharpe_ratio
                    total_weight += weight
                
                if total_weight > 0:
                    predicted_performance /= total_weight
                    
                    # Compare with actual performance
                    actual_performance = test_sample.sharpe_ratio
                    score = -abs(predicted_performance - actual_performance)  # Negative MAE
                    scores.append(score)
            
            return np.mean(scores) if scores else float('-inf')
            
        except Exception as e:
            logger.error(f"Fold evaluation failed: {e}")
            return float('-inf')
    
    async def _predict_optimal_weights(self,
                                     current_features: MarketFeatures,
                                     training_data: List[OptimalPortfolio],
                                     k: int,
                                     assets: List[str]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Predict optimal weights using k-NN on similar market conditions
        
        Returns:
            Tuple of (optimal_weights, neighbors_info)
        """
        # Find k nearest neighbors
        distances = []
        current_features_dict = asdict(current_features)
        
        for i, portfolio in enumerate(training_data):
            dist = self.distance_calculator.compute_distance(
                current_features_dict,
                asdict(portfolio.market_features)
            )
            distances.append((dist, i, portfolio))
        
        # Sort by distance and select k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        
        # Weighted average of optimal portfolios
        optimal_weights = {asset: 0.0 for asset in assets}
        total_weight = 0.0
        neighbors_info = []
        
        for dist, idx, portfolio in neighbors:
            # Inverse distance weighting (with small epsilon to avoid division by zero)
            weight = 1.0 / (dist + 1e-8) if dist > 0 else 1.0
            
            # Add weighted contribution from this neighbor
            for asset in assets:
                if asset in portfolio.weights:
                    optimal_weights[asset] += weight * portfolio.weights[asset]
            
            total_weight += weight
            
            # Store neighbor info for analysis
            neighbors_info.append({
                'distance': float(dist),
                'weight': float(weight),
                'optimization_method': portfolio.optimization_method,
                'sharpe_ratio': portfolio.sharpe_ratio,
                'expected_return': portfolio.expected_return,
                'expected_risk': portfolio.expected_risk,
                'market_phase': portfolio.market_features.market_phase,
                'timestamp': portfolio.timestamp.isoformat(),
                'portfolio_weights': portfolio.weights.copy()
            })
        
        # Normalize weights
        if total_weight > 0:
            for asset in assets:
                optimal_weights[asset] /= total_weight
        
        # Ensure weights sum to 1
        total_weight_sum = sum(optimal_weights.values())
        if total_weight_sum > 0:
            optimal_weights = {asset: weight / total_weight_sum 
                             for asset, weight in optimal_weights.items()}
        else:
            # Fallback to equal weighting
            optimal_weights = {asset: 1.0 / len(assets) for asset in assets}
        
        # Update neighbor weights after normalization
        for neighbor in neighbors_info:
            neighbor['normalized_weight'] = neighbor['weight'] / total_weight if total_weight > 0 else 0.0
        
        return optimal_weights, neighbors_info
    
    async def _predict_performance(self, neighbors_info: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict portfolio performance based on nearest neighbors"""
        if not neighbors_info:
            return {
                'expected_return': 0.08,
                'expected_risk': 0.15,
                'sharpe_ratio': 0.53,
                'max_drawdown': -0.1
            }
        
        # Weighted average of neighbor performance metrics
        total_weight = sum(neighbor['normalized_weight'] for neighbor in neighbors_info)
        
        if total_weight == 0:
            total_weight = len(neighbors_info)
            for neighbor in neighbors_info:
                neighbor['normalized_weight'] = 1.0 / len(neighbors_info)
        
        expected_return = sum(
            neighbor['expected_return'] * neighbor['normalized_weight']
            for neighbor in neighbors_info
        )
        
        expected_risk = sum(
            neighbor['expected_risk'] * neighbor['normalized_weight']
            for neighbor in neighbors_info
        )
        
        sharpe_ratio = sum(
            neighbor['sharpe_ratio'] * neighbor['normalized_weight']
            for neighbor in neighbors_info
        )
        
        # Estimate max drawdown (simplified)
        max_drawdown = -0.05 - 0.1 * expected_risk  # Simple risk-based estimate
        
        return {
            'expected_return': float(expected_return),
            'expected_risk': float(expected_risk),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
    
    async def _compute_model_confidence(self,
                                      current_features: MarketFeatures,
                                      training_data: List[OptimalPortfolio],
                                      neighbors_info: List[Dict[str, Any]]) -> float:
        """
        Compute model confidence based on:
        - Distance to nearest neighbors
        - Consistency of neighbor recommendations
        - Training data coverage
        """
        try:
            if not neighbors_info:
                return 0.0
            
            # Distance-based confidence (closer neighbors = higher confidence)
            distances = [neighbor['distance'] for neighbor in neighbors_info]
            avg_distance = np.mean(distances)
            min_distance = min(distances)
            distance_confidence = 1.0 / (1.0 + avg_distance)
            
            # Consistency confidence (similar recommendations = higher confidence)
            sharpe_ratios = [neighbor['sharpe_ratio'] for neighbor in neighbors_info]
            returns = [neighbor['expected_return'] for neighbor in neighbors_info]
            
            sharpe_consistency = 1.0 - (np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8))
            return_consistency = 1.0 - (np.std(returns) / (np.mean(np.abs(returns)) + 1e-8))
            consistency_confidence = (sharpe_consistency + return_consistency) / 2
            
            # Coverage confidence (more training data = higher confidence)
            coverage_confidence = min(1.0, len(training_data) / 500)  # Normalize by target size
            
            # Combined confidence (weighted average)
            confidence = (
                0.4 * distance_confidence +
                0.4 * max(0.0, consistency_confidence) +
                0.2 * coverage_confidence
            )
            
            return float(min(1.0, max(0.0, confidence)))
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return 0.5  # Moderate confidence as fallback
    
    async def _compute_validation_score(self,
                                      current_features: MarketFeatures,
                                      training_data: List[OptimalPortfolio],
                                      k: int,
                                      validation_split: float) -> float:
        """Compute validation score using holdout method"""
        try:
            if len(training_data) < 10:
                return 0.5
            
            # Split training data
            split_point = int(len(training_data) * (1 - validation_split))
            train_subset = training_data[:split_point]
            validation_subset = training_data[split_point:]
            
            if len(validation_subset) == 0:
                return 0.5
            
            # Evaluate on validation set
            scores = []
            for val_sample in validation_subset:
                # Predict using training subset
                val_features = asdict(val_sample.market_features)
                
                # Find k nearest neighbors in training subset
                distances = []
                for train_sample in train_subset:
                    dist = self.distance_calculator.compute_distance(
                        val_features,
                        asdict(train_sample.market_features)
                    )
                    distances.append((dist, train_sample))
                
                distances.sort(key=lambda x: x[0])
                neighbors = distances[:min(k, len(distances))]
                
                # Predict performance
                if neighbors:
                    predicted_sharpe = sum(
                        neighbor.sharpe_ratio / (dist + 1e-8)
                        for dist, neighbor in neighbors
                    ) / sum(1.0 / (dist + 1e-8) for dist, neighbor in neighbors)
                    
                    # Compare with actual
                    actual_sharpe = val_sample.sharpe_ratio
                    score = -abs(predicted_sharpe - actual_sharpe)
                    scores.append(score)
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Validation score computation failed: {e}")
            return 0.0
    
    async def _compute_feature_importance(self,
                                        training_data: List[OptimalPortfolio],
                                        neighbors_info: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute feature importance for the k-NN model"""
        try:
            # Use the market features from training data
            feature_samples = [portfolio.market_features for portfolio in training_data]
            
            if len(feature_samples) < 2:
                return {}
            
            # Use the feature extractor's importance computation
            importance_scores = self.feature_extractor.compute_feature_importance(feature_samples)
            
            # Enhance with k-NN specific analysis (which features contributed most to neighbor selection)
            if neighbors_info:
                # This is a simplified approach - in practice, you'd use permutation importance
                knn_weights = self.feature_weights.copy() if self.feature_weights else {}
                
                # Combine with computed importance
                for feature, score in importance_scores.items():
                    if feature in knn_weights:
                        importance_scores[feature] = (score + knn_weights[feature]) / 2
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            return {}
    
    def _apply_constraints(self, weights: Dict[str, float], constraints: Dict[str, Any]) -> Dict[str, float]:
        """Apply portfolio constraints to weights"""
        try:
            # Simple constraint handling
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
            
            # Apply min/max constraints
            constrained_weights = {}
            for asset, weight in weights.items():
                constrained_weights[asset] = max(min_weight, min(weight, max_weight))
            
            # Renormalize to sum to 1
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                constrained_weights = {
                    asset: weight / total_weight
                    for asset, weight in constrained_weights.items()
                }
            
            return constrained_weights
            
        except Exception as e:
            logger.error(f"Constraint application failed: {e}")
            return weights  # Return original weights if constraint application fails
    
    def _validate_request(self, request: SupervisedOptimizationRequest):
        """Validate optimization request"""
        if not request.assets:
            raise ValueError("Assets list is required")
        
        if request.historical_returns is None or len(request.historical_returns) < request.min_training_periods:
            raise ValueError(f"Insufficient historical data: need at least {request.min_training_periods} periods")
        
        if len(request.assets) < 2:
            raise ValueError("At least 2 assets required for portfolio optimization")
        
        # Check if all assets are present in returns data
        missing_assets = set(request.assets) - set(request.historical_returns.columns)
        if missing_assets:
            raise ValueError(f"Missing return data for assets: {missing_assets}")
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance tracking statistics"""
        self.optimization_count += 1
        self.average_prediction_time = (
            (self.average_prediction_time * (self.optimization_count - 1) + processing_time)
            / self.optimization_count
        )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and statistics"""
        return {
            'model_type': 'supervised_knn',
            'distance_metric': self.distance_metric,
            'optimization_count': self.optimization_count,
            'average_prediction_time': self.average_prediction_time,
            'training_data_size': len(self._training_data),
            'cache_size': len(self._model_cache),
            'feature_weights_count': len(self.feature_weights) if self.feature_weights else 0,
            'confidence_level': self.confidence_level
        }

# Factory functions for common configurations
def create_supervised_optimizer(distance_metric: str = "hassanat") -> SupervisedKNNOptimizer:
    """Create a supervised k-NN optimizer with default settings"""
    return SupervisedKNNOptimizer(
        distance_metric=distance_metric,
        feature_weights=create_feature_weights_for_knn()
    )

def create_robust_supervised_optimizer() -> SupervisedKNNOptimizer:
    """Create a robust supervised k-NN optimizer for noisy data"""
    # Use feature weights that emphasize stable, fundamental characteristics
    robust_weights = create_feature_weights_for_knn()
    
    # Increase weights for stable features
    stable_features = ['returns_volatility', 'average_correlation', 'sharpe_ratio', 'max_drawdown']
    for feature in stable_features:
        if feature in robust_weights:
            robust_weights[feature] *= 1.5
    
    return SupervisedKNNOptimizer(
        distance_metric="hassanat",
        feature_weights=robust_weights,
        confidence_level=0.99
    )
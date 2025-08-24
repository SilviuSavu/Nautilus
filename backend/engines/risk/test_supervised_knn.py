#!/usr/bin/env python3
"""
Comprehensive Test Suite for Supervised k-NN Portfolio Optimization
Tests mathematical correctness, performance, and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any
import logging

from distance_metrics import (
    DistanceCalculator, DistanceConfig, DistanceMetric, 
    create_distance_calculator, get_default_financial_calculator
)
from market_features import (
    MarketFeatureExtractor, MarketFeatures, extract_market_features,
    create_feature_weights_for_knn
)
from supervised_knn_optimizer import (
    SupervisedKNNOptimizer, SupervisedOptimizationRequest,
    create_supervised_optimizer, create_robust_supervised_optimizer
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDistanceMetrics:
    """Test suite for distance metric calculations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.calculator = create_distance_calculator("hassanat", True)
        self.sample_features_1 = {
            'returns_volatility': 0.15,
            'sharpe_ratio': 1.2,
            'correlation': 0.6,
            'momentum_1m': 0.02
        }
        self.sample_features_2 = {
            'returns_volatility': 0.18,
            'sharpe_ratio': 1.0,
            'correlation': 0.7,
            'momentum_1m': 0.015
        }
    
    def test_hassanat_distance_calculation(self):
        """Test Hassanat distance calculation correctness"""
        distance = self.calculator.compute_distance(
            self.sample_features_1, 
            self.sample_features_2
        )
        
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert not np.isnan(distance)
        assert not np.isinf(distance)
    
    def test_distance_symmetry(self):
        """Test that distance is symmetric: d(A,B) = d(B,A)"""
        dist_1_2 = self.calculator.compute_distance(
            self.sample_features_1, self.sample_features_2
        )
        dist_2_1 = self.calculator.compute_distance(
            self.sample_features_2, self.sample_features_1
        )
        
        assert abs(dist_1_2 - dist_2_1) < 1e-10
    
    def test_distance_identity(self):
        """Test that distance to self is zero"""
        distance = self.calculator.compute_distance(
            self.sample_features_1, self.sample_features_1
        )
        
        assert distance == 0.0
    
    def test_triangle_inequality(self):
        """Test triangle inequality: d(A,C) <= d(A,B) + d(B,C)"""
        features_3 = {
            'returns_volatility': 0.20,
            'sharpe_ratio': 0.8,
            'correlation': 0.8,
            'momentum_1m': 0.01
        }
        
        dist_1_2 = self.calculator.compute_distance(self.sample_features_1, self.sample_features_2)
        dist_2_3 = self.calculator.compute_distance(self.sample_features_2, features_3)
        dist_1_3 = self.calculator.compute_distance(self.sample_features_1, features_3)
        
        assert dist_1_3 <= dist_1_2 + dist_2_3 + 1e-10  # Small epsilon for numerical precision
    
    def test_multiple_distance_metrics(self):
        """Test different distance metrics produce reasonable results"""
        metrics = ["hassanat", "euclidean", "manhattan", "cosine"]
        distances = {}
        
        for metric in metrics:
            calc = create_distance_calculator(metric)
            distances[metric] = calc.compute_distance(
                self.sample_features_1, self.sample_features_2
            )
        
        # All distances should be positive and finite
        for metric, distance in distances.items():
            assert distance >= 0.0
            assert not np.isnan(distance)
            assert not np.isinf(distance)
        
        # Hassanat should be scale-invariant (generally different from Euclidean)
        assert distances["hassanat"] != distances["euclidean"]
    
    def test_missing_value_handling(self):
        """Test handling of missing values in features"""
        features_with_nan = {
            'returns_volatility': np.nan,
            'sharpe_ratio': 1.2,
            'correlation': 0.6,
            'momentum_1m': 0.02
        }
        
        # Should not raise exception and return finite distance
        distance = self.calculator.compute_distance(
            features_with_nan, self.sample_features_2
        )
        
        assert isinstance(distance, float)
        assert not np.isnan(distance)
        assert not np.isinf(distance)
    
    def test_extreme_values(self):
        """Test behavior with extreme feature values"""
        extreme_features = {
            'returns_volatility': 10.0,  # Very high volatility
            'sharpe_ratio': -5.0,        # Very negative Sharpe
            'correlation': 1.0,          # Perfect correlation
            'momentum_1m': 0.5           # Extreme momentum
        }
        
        distance = self.calculator.compute_distance(
            extreme_features, self.sample_features_1
        )
        
        assert isinstance(distance, float)
        assert not np.isnan(distance)
        assert not np.isinf(distance)

class TestMarketFeatures:
    """Test suite for market feature extraction"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = MarketFeatureExtractor()
        
        # Create realistic sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        self.sample_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.0008, 0.02, 252),
            'GOOGL': np.random.normal(0.001, 0.025, 252),
            'MSFT': np.random.normal(0.0007, 0.018, 252)
        }, index=dates)
    
    def test_basic_feature_extraction(self):
        """Test basic feature extraction functionality"""
        features = self.extractor.extract_features(self.sample_returns)
        
        assert isinstance(features, MarketFeatures)
        assert hasattr(features, 'returns_volatility')
        assert hasattr(features, 'sharpe_ratio')
        assert hasattr(features, 'average_correlation')
        assert hasattr(features, 'momentum_1m')
    
    def test_feature_value_ranges(self):
        """Test that extracted features are in reasonable ranges"""
        features = self.extractor.extract_features(self.sample_returns)
        
        # Volatility should be positive
        assert features.returns_volatility > 0
        
        # Correlation should be between -1 and 1
        assert -1 <= features.average_correlation <= 1
        
        # Max eigenvalue should be positive
        assert features.max_eigenvalue > 0
        
        # Bull/bear indicator should be between -1 and 1
        assert -1 <= features.bull_bear_indicator <= 1
    
    def test_feature_consistency(self):
        """Test that features are consistent across multiple extractions"""
        features_1 = self.extractor.extract_features(self.sample_returns)
        features_2 = self.extractor.extract_features(self.sample_returns)
        
        # Same data should produce same features
        assert abs(features_1.returns_volatility - features_2.returns_volatility) < 1e-10
        assert abs(features_1.sharpe_ratio - features_2.sharpe_ratio) < 1e-10
        assert abs(features_1.average_correlation - features_2.average_correlation) < 1e-10
    
    def test_minimal_data_handling(self):
        """Test feature extraction with minimal data"""
        # Very short time series
        short_returns = self.sample_returns.tail(30)
        
        features = self.extractor.extract_features(short_returns)
        
        # Should still extract features without errors
        assert isinstance(features, MarketFeatures)
        assert not np.isnan(features.returns_volatility)
        assert not np.isnan(features.sharpe_ratio)
    
    def test_market_phase_classification(self):
        """Test market phase classification logic"""
        # Create bull market data
        bull_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.002, 0.015, 100),  # Higher returns, lower vol
            'GOOGL': np.random.normal(0.0025, 0.018, 100)
        })
        
        bull_features = self.extractor.extract_features(bull_returns)
        
        # Should classify as bullish (though simplified test)
        assert bull_features.market_phase in ['bull', 'volatile_bull', 'sideways']
        
        # Create bear market data
        bear_returns = pd.DataFrame({
            'AAPL': np.random.normal(-0.002, 0.025, 100),  # Negative returns, higher vol
            'GOOGL': np.random.normal(-0.0015, 0.03, 100)
        })
        
        bear_features = self.extractor.extract_features(bear_returns)
        
        # Should classify as bearish or volatile
        assert bear_features.market_phase in ['bear', 'volatile_bear', 'volatile', 'sideways']

class TestSupervisedOptimizer:
    """Test suite for supervised k-NN optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = create_supervised_optimizer()
        
        # Create realistic test data
        np.random.seed(42)
        self.assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Generate correlated returns
        n_assets = len(self.assets)
        correlation = 0.3
        cov_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov_matrix, 1.0)
        
        # Scale by different volatilities
        vols = np.array([0.25, 0.30, 0.20, 0.35])
        cov_matrix = np.outer(vols, vols) * cov_matrix
        
        returns_data = np.random.multivariate_normal(
            mean=[0.0008, 0.001, 0.0006, 0.0012],
            cov=cov_matrix,
            size=len(dates)
        )
        
        self.historical_returns = pd.DataFrame(
            returns_data, index=dates, columns=self.assets
        )
    
    @pytest.mark.asyncio
    async def test_basic_optimization(self):
        """Test basic supervised optimization functionality"""
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=self.historical_returns,
            k_neighbors=5,
            min_training_periods=100,  # Reduced for testing
            cross_validation_folds=3   # Reduced for testing
        )
        
        result = await self.optimizer.optimize_portfolio(request)
        
        # Basic result validation
        assert hasattr(result, 'optimal_weights')
        assert hasattr(result, 'k_neighbors_used')
        assert hasattr(result, 'model_confidence')
        
        # Weight validation
        weights = result.optimal_weights
        assert len(weights) == len(self.assets)
        assert all(asset in weights for asset in self.assets)
        assert all(weight >= 0 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-10
    
    @pytest.mark.asyncio
    async def test_dynamic_k_selection(self):
        """Test dynamic k selection via cross-validation"""
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=self.historical_returns,
            k_neighbors=None,  # Use dynamic selection
            min_training_periods=100,
            cross_validation_folds=3
        )
        
        result = await self.optimizer.optimize_portfolio(request)
        
        # k should be selected automatically
        assert result.k_neighbors_used > 0
        assert result.k_neighbors_used <= 50  # Reasonable upper bound
        
        # Should have some confidence in the result
        assert 0 <= result.model_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_distance_metric_consistency(self):
        """Test that different distance metrics produce consistent results"""
        metrics = ["hassanat", "euclidean", "manhattan"]
        results = {}
        
        for metric in metrics:
            optimizer = SupervisedKNNOptimizer(distance_metric=metric)
            request = SupervisedOptimizationRequest(
                assets=self.assets,
                historical_returns=self.historical_returns,
                k_neighbors=5,
                distance_metric=metric,
                min_training_periods=100,
                cross_validation_folds=3
            )
            
            results[metric] = await optimizer.optimize_portfolio(request)
        
        # All should produce valid results
        for metric, result in results.items():
            assert sum(result.optimal_weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert result.model_confidence >= 0
            assert result.k_neighbors_used > 0
        
        # Results should be different (different metrics capture different similarities)
        hassanat_weights = list(results["hassanat"].optimal_weights.values())
        euclidean_weights = list(results["euclidean"].optimal_weights.values())
        
        # At least some weights should differ
        assert not np.allclose(hassanat_weights, euclidean_weights, atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_constraint_application(self):
        """Test that portfolio constraints are properly applied"""
        constraints = {
            'min_weight': 0.1,
            'max_weight': 0.4
        }
        
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=self.historical_returns,
            k_neighbors=5,
            constraints=constraints,
            min_training_periods=100,
            cross_validation_folds=3
        )
        
        result = await self.optimizer.optimize_portfolio(request)
        
        # Check constraints are satisfied
        for weight in result.optimal_weights.values():
            assert weight >= constraints['min_weight'] - 1e-6
            assert weight <= constraints['max_weight'] + 1e-6
        
        # Still sum to 1
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient historical data"""
        # Very short historical data
        short_returns = self.historical_returns.tail(50)
        
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=short_returns,
            min_training_periods=200  # More than available data
        )
        
        # Should raise appropriate error
        with pytest.raises(ValueError, match="Insufficient"):
            await self.optimizer.optimize_portfolio(request)
    
    @pytest.mark.asyncio
    async def test_performance_prediction(self):
        """Test that performance predictions are reasonable"""
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=self.historical_returns,
            k_neighbors=7,
            min_training_periods=100,
            cross_validation_folds=3
        )
        
        result = await self.optimizer.optimize_portfolio(request)
        
        # Performance predictions should be reasonable
        assert -1.0 <= result.expected_return <= 1.0  # Annual return
        assert 0.0 <= result.expected_risk <= 2.0     # Annual volatility
        assert -5.0 <= result.sharpe_ratio <= 5.0     # Sharpe ratio
        
        # Should have metadata
        assert len(result.nearest_neighbors_info) > 0
        assert len(result.feature_importance) > 0
        assert result.training_periods > 0

class TestIntegrationAndPerformance:
    """Integration tests and performance validation"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.optimizer = create_supervised_optimizer()
        
        # Create larger, more realistic dataset
        np.random.seed(42)
        self.assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        # 5 years of daily data
        dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
        n_assets = len(self.assets)
        
        # More sophisticated return generation with regime changes
        returns_list = []
        for i, date in enumerate(dates):
            # Create regime-dependent returns
            if i < len(dates) * 0.7:  # Bull market period
                mean_returns = np.random.normal(0.001, 0.0002, n_assets)
                vol_scaling = 0.8  # Lower volatility in bull market
            else:  # Bear/volatile market period
                mean_returns = np.random.normal(-0.0005, 0.0003, n_assets)
                vol_scaling = 1.3  # Higher volatility in bear market
            
            # Generate returns with time-varying correlation
            base_corr = 0.3 + 0.2 * np.sin(2 * np.pi * i / 252)  # Annual cycle
            cov_matrix = np.full((n_assets, n_assets), base_corr)
            np.fill_diagonal(cov_matrix, 1.0)
            
            daily_vols = np.random.uniform(0.015, 0.035, n_assets) * vol_scaling
            cov_matrix = np.outer(daily_vols, daily_vols) * cov_matrix
            
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            returns_list.append(daily_returns)
        
        self.historical_returns = pd.DataFrame(
            returns_list, index=dates, columns=self.assets
        )
    
    @pytest.mark.asyncio
    async def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline with realistic data"""
        request = SupervisedOptimizationRequest(
            assets=self.assets,
            historical_returns=self.historical_returns,
            k_neighbors=None,  # Dynamic selection
            distance_metric="hassanat",
            lookback_periods=252,
            min_training_periods=504,  # 2 years
            validation_split=0.2,
            cross_validation_folds=5
        )
        
        start_time = datetime.now()
        result = await self.optimizer.optimize_portfolio(request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Performance requirements
        assert processing_time < 60.0  # Should complete within 60 seconds
        
        # Quality requirements
        assert result.model_confidence > 0.3  # Reasonable confidence
        assert result.k_neighbors_used >= 3   # Use multiple neighbors
        assert result.training_periods >= 50  # Sufficient training data
        
        # Portfolio quality
        weights = result.optimal_weights
        assert len(weights) == len(self.assets)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(0 <= weight <= 1 for weight in weights.values())
        
        # Diversification check (no single asset > 50%)
        max_weight = max(weights.values())
        assert max_weight <= 0.5
        
        logger.info(f"Full pipeline test completed in {processing_time:.2f} seconds")
        logger.info(f"Optimal weights: {weights}")
        logger.info(f"Model confidence: {result.model_confidence:.3f}")
        logger.info(f"K neighbors used: {result.k_neighbors_used}")
    
    @pytest.mark.asyncio
    async def test_performance_vs_equal_weight(self):
        """Test that supervised optimization outperforms equal weighting"""
        request = SupervisedOptimizationRequest(
            assets=self.assets[:4],  # Use fewer assets for faster testing
            historical_returns=self.historical_returns[self.assets[:4]],
            k_neighbors=5,
            min_training_periods=252,
            cross_validation_folds=3
        )
        
        result = await self.optimizer.optimize_portfolio(request)
        
        # Compare with equal weight benchmark
        equal_weights = {asset: 0.25 for asset in self.assets[:4]}
        
        # Calculate equal weight performance on same data
        recent_returns = self.historical_returns[self.assets[:4]].tail(252)
        equal_weight_returns = (recent_returns * 0.25).sum(axis=1)
        equal_weight_sharpe = (equal_weight_returns.mean() / equal_weight_returns.std()) * np.sqrt(252)
        
        logger.info(f"Supervised Sharpe: {result.sharpe_ratio:.3f}")
        logger.info(f"Equal weight Sharpe: {equal_weight_sharpe:.3f}")
        
        # Supervised method should be competitive (allowing for noise in test data)
        # This is a statistical test, so we use a reasonable tolerance
        assert result.sharpe_ratio >= equal_weight_sharpe - 0.5
    
    def test_mathematical_properties(self):
        """Test mathematical properties and invariants"""
        # Test feature weight creation
        feature_weights = create_feature_weights_for_knn()
        
        assert isinstance(feature_weights, dict)
        assert all(weight > 0 for weight in feature_weights.values())
        assert 'returns_volatility' in feature_weights
        assert 'sharpe_ratio' in feature_weights
        
        # Test distance calculator configurations
        robust_calc = get_default_financial_calculator()
        assert isinstance(robust_calc, DistanceCalculator)
        
        # Test optimizer factory functions
        basic_optimizer = create_supervised_optimizer()
        robust_optimizer = create_robust_supervised_optimizer()
        
        assert isinstance(basic_optimizer, SupervisedKNNOptimizer)
        assert isinstance(robust_optimizer, SupervisedKNNOptimizer)
        assert basic_optimizer.distance_metric == "hassanat"
        assert robust_optimizer.distance_metric == "hassanat"

class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness"""
    
    @pytest.mark.asyncio
    async def test_single_asset_handling(self):
        """Test handling of edge case with too few assets"""
        optimizer = create_supervised_optimizer()
        
        # Single asset (should fail)
        single_asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100)
        })
        
        request = SupervisedOptimizationRequest(
            assets=['AAPL'],
            historical_returns=single_asset_returns
        )
        
        with pytest.raises(ValueError, match="At least 2 assets"):
            await optimizer.optimize_portfolio(request)
    
    @pytest.mark.asyncio
    async def test_missing_asset_data(self):
        """Test handling of missing asset data"""
        optimizer = create_supervised_optimizer()
        
        # Returns data missing one asset
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.025, 100)
        })
        
        request = SupervisedOptimizationRequest(
            assets=['AAPL', 'GOOGL', 'MSFT'],  # MSFT missing from returns
            historical_returns=returns
        )
        
        with pytest.raises(ValueError, match="Missing return data"):
            await optimizer.optimize_portfolio(request)
    
    def test_extreme_correlation_cases(self):
        """Test handling of extreme correlation scenarios"""
        # Perfect correlation case
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, 100)
        
        perfect_corr_returns = pd.DataFrame({
            'AAPL': base_returns,
            'GOOGL': base_returns,  # Identical returns
            'MSFT': base_returns * 1.1  # Scaled but perfectly correlated
        })
        
        extractor = MarketFeatureExtractor()
        features = extractor.extract_features(perfect_corr_returns)
        
        # Should handle perfect correlation gracefully
        assert not np.isnan(features.average_correlation)
        assert not np.isnan(features.max_eigenvalue)
    
    def test_zero_variance_assets(self):
        """Test handling of zero-variance assets"""
        # Create returns with one constant asset
        returns_data = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'BOND': np.zeros(100),  # Zero variance
            'GOOGL': np.random.normal(0.001, 0.025, 100)
        }
        
        returns = pd.DataFrame(returns_data)
        extractor = MarketFeatureExtractor()
        
        # Should not crash on zero variance
        features = extractor.extract_features(returns)
        
        assert not np.isnan(features.returns_volatility)
        assert not np.isinf(features.returns_volatility)

# Benchmarking utilities
class PerformanceBenchmark:
    """Utilities for performance benchmarking"""
    
    @staticmethod
    async def benchmark_optimization_speed(optimizer: SupervisedKNNOptimizer, 
                                         n_assets: int = 5, 
                                         n_periods: int = 1000,
                                         n_trials: int = 3) -> Dict[str, float]:
        """Benchmark optimization speed"""
        # Generate test data
        np.random.seed(42)
        assets = [f'ASSET_{i}' for i in range(n_assets)]
        
        returns_data = np.random.multivariate_normal(
            mean=np.random.normal(0.001, 0.0002, n_assets),
            cov=np.random.uniform(0.0001, 0.0004, (n_assets, n_assets)),
            size=n_periods
        )
        
        returns_df = pd.DataFrame(returns_data, columns=assets)
        
        # Run benchmark trials
        times = []
        for trial in range(n_trials):
            request = SupervisedOptimizationRequest(
                assets=assets,
                historical_returns=returns_df,
                k_neighbors=5,
                min_training_periods=min(252, n_periods // 2),
                cross_validation_folds=3
            )
            
            start_time = datetime.now()
            await optimizer.optimize_portfolio(request)
            duration = (datetime.now() - start_time).total_seconds()
            times.append(duration)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'n_assets': n_assets,
            'n_periods': n_periods,
            'n_trials': n_trials
        }

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_basic"  # Run basic tests only for quick validation
    ])
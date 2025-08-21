"""
Unit tests for risk calculator functionality
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock
from risk_calculator import (
    VaRCalculator, 
    CorrelationCalculator, 
    RiskMetricsCalculator,
    RiskCalculator
)

class TestVaRCalculator:
    """Test VaR calculation methods"""
    
    @pytest.fixture
    def var_calculator(self):
        return VaRCalculator()
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    
    @pytest.mark.asyncio
    async def test_historical_var_basic(self, var_calculator, sample_returns):
        """Test basic historical VaR calculation"""
        var_95 = await var_calculator.historical_var(sample_returns, 0.95, 1)
        var_99 = await var_calculator.historical_var(sample_returns, 0.99, 1)
        
        assert var_95 > 0
        assert var_99 > 0
        assert var_99 > var_95  # 99% VaR should be higher than 95% VaR
        assert isinstance(var_95, float)
    
    @pytest.mark.asyncio
    async def test_historical_var_time_horizons(self, var_calculator, sample_returns):
        """Test VaR scaling with time horizons"""
        var_1d = await var_calculator.historical_var(sample_returns, 0.95, 1)
        var_7d = await var_calculator.historical_var(sample_returns, 0.95, 7)
        var_30d = await var_calculator.historical_var(sample_returns, 0.95, 30)
        
        # Longer time horizons should generally have higher VaR
        assert var_7d > var_1d
        assert var_30d > var_7d
    
    @pytest.mark.asyncio
    async def test_historical_var_insufficient_data(self, var_calculator):
        """Test VaR calculation with insufficient data"""
        short_returns = np.random.normal(0, 0.02, 10)  # Only 10 observations
        
        # Should still calculate but warn
        var = await var_calculator.historical_var(short_returns, 0.95, 1)
        assert var > 0
    
    @pytest.mark.asyncio
    async def test_parametric_var_normal(self, var_calculator, sample_returns):
        """Test parametric VaR with normal distribution"""
        var_95 = await var_calculator.parametric_var(sample_returns, 0.95, 1, 'normal')
        var_99 = await var_calculator.parametric_var(sample_returns, 0.99, 1, 'normal')
        
        assert var_95 > 0
        assert var_99 > 0
        assert var_99 > var_95
    
    @pytest.mark.asyncio
    async def test_parametric_var_t_student(self, var_calculator, sample_returns):
        """Test parametric VaR with t-student distribution"""
        var_t = await var_calculator.parametric_var(sample_returns, 0.95, 1, 't_student')
        assert var_t > 0
    
    @pytest.mark.asyncio
    async def test_monte_carlo_var(self, var_calculator, sample_returns):
        """Test Monte Carlo VaR calculation"""
        var_mc = await var_calculator.monte_carlo_var(
            sample_returns, 0.95, 1, num_simulations=1000, random_seed=42
        )
        assert var_mc > 0
        
        # Test reproducibility with same seed
        var_mc2 = await var_calculator.monte_carlo_var(
            sample_returns, 0.95, 1, num_simulations=1000, random_seed=42
        )
        assert abs(var_mc - var_mc2) < 1e-10  # Should be identical
    
    @pytest.mark.asyncio
    async def test_expected_shortfall(self, var_calculator, sample_returns):
        """Test Expected Shortfall calculation"""
        es_95 = await var_calculator.expected_shortfall(sample_returns, 0.95, 1)
        es_99 = await var_calculator.expected_shortfall(sample_returns, 0.99, 1)
        
        assert es_95 > 0
        assert es_99 > 0
        assert es_99 > es_95  # ES should be higher for higher confidence
    
    @pytest.mark.asyncio
    async def test_var_with_nans(self, var_calculator):
        """Test VaR calculation with NaN values"""
        returns_with_nans = np.array([0.01, np.nan, -0.02, 0.005, np.nan, -0.01] * 10)
        
        var = await var_calculator.historical_var(returns_with_nans, 0.95, 1)
        assert var > 0
        assert np.isfinite(var)

class TestCorrelationCalculator:
    """Test correlation calculation methods"""
    
    @pytest.fixture
    def corr_calculator(self):
        return CorrelationCalculator()
    
    @pytest.fixture
    def sample_returns_data(self):
        """Generate sample multi-asset return data"""
        np.random.seed(42)
        n_obs = 252
        
        # Create correlated returns
        base_returns = np.random.normal(0.001, 0.02, n_obs)
        
        returns_data = {
            'AAPL': base_returns + np.random.normal(0, 0.01, n_obs),
            'GOOGL': base_returns * 0.8 + np.random.normal(0, 0.015, n_obs),
            'MSFT': base_returns * 0.6 + np.random.normal(0, 0.012, n_obs),
            'TSLA': np.random.normal(0.002, 0.04, n_obs)  # Less correlated
        }
        
        return returns_data
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_pearson(self, corr_calculator, sample_returns_data):
        """Test Pearson correlation matrix calculation"""
        corr_matrix, symbols = await corr_calculator.correlation_matrix(
            sample_returns_data, method='pearson'
        )
        
        assert corr_matrix.shape == (4, 4)
        assert len(symbols) == 4
        assert all(symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA'] for symbol in symbols)
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(4))
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
        
        # All correlations should be between -1 and 1
        assert np.all(corr_matrix >= -1)
        assert np.all(corr_matrix <= 1)
    
    @pytest.mark.asyncio
    async def test_correlation_matrix_ledoit_wolf(self, corr_calculator, sample_returns_data):
        """Test Ledoit-Wolf shrinkage correlation matrix"""
        corr_matrix, symbols = await corr_calculator.correlation_matrix(
            sample_returns_data, method='ledoit_wolf'
        )
        
        assert corr_matrix.shape == (4, 4)
        assert len(symbols) == 4
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(4))
    
    @pytest.mark.asyncio
    async def test_rolling_correlation(self, corr_calculator):
        """Test rolling correlation calculation"""
        np.random.seed(42)
        returns1 = np.random.normal(0.001, 0.02, 100)
        returns2 = returns1 * 0.8 + np.random.normal(0, 0.01, 100)
        
        rolling_corr = await corr_calculator.rolling_correlation(
            returns1, returns2, window=30
        )
        
        assert len(rolling_corr) == 100
        # First 29 values should be NaN due to window
        assert np.isnan(rolling_corr[:29]).all()
        # Remaining values should be finite
        assert np.isfinite(rolling_corr[29:]).all()
    
    @pytest.mark.asyncio
    async def test_portfolio_beta(self, corr_calculator):
        """Test portfolio beta calculation"""
        np.random.seed(42)
        market_returns = np.random.normal(0.001, 0.015, 252)
        portfolio_returns = market_returns * 1.2 + np.random.normal(0, 0.01, 252)
        
        beta, alpha, r_squared = await corr_calculator.portfolio_beta(
            portfolio_returns, market_returns
        )
        
        assert isinstance(beta, float)
        assert isinstance(alpha, float)
        assert isinstance(r_squared, float)
        
        # Beta should be close to 1.2 (our construction)
        assert 1.0 < beta < 1.4
        assert 0 <= r_squared <= 1

class TestRiskMetricsCalculator:
    """Test portfolio risk metrics"""
    
    @pytest.fixture
    def metrics_calculator(self):
        return RiskMetricsCalculator()
    
    @pytest.fixture
    def sample_portfolio_returns(self):
        """Generate sample portfolio returns"""
        np.random.seed(42)
        return np.random.normal(0.0008, 0.015, 252)  # Daily returns
    
    @pytest.mark.asyncio
    async def test_portfolio_volatility(self, metrics_calculator, sample_portfolio_returns):
        """Test portfolio volatility calculation"""
        vol_daily = await metrics_calculator.portfolio_volatility(
            sample_portfolio_returns, annualized=False
        )
        vol_annual = await metrics_calculator.portfolio_volatility(
            sample_portfolio_returns, annualized=True
        )
        
        assert vol_daily > 0
        assert vol_annual > 0
        assert vol_annual > vol_daily  # Annualized should be higher
        
        # Check approximate scaling
        expected_annual = vol_daily * np.sqrt(252)
        assert abs(vol_annual - expected_annual) < 1e-10
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio(self, metrics_calculator, sample_portfolio_returns):
        """Test Sharpe ratio calculation"""
        sharpe = await metrics_calculator.sharpe_ratio(
            sample_portfolio_returns, risk_free_rate=0.02
        )
        
        assert isinstance(sharpe, float)
        # With our constructed data, Sharpe should be reasonable
        assert -2 < sharpe < 2
    
    @pytest.mark.asyncio
    async def test_maximum_drawdown(self, metrics_calculator):
        """Test maximum drawdown calculation"""
        # Create returns with known drawdown
        returns = np.array([0.05, -0.10, -0.05, 0.03, 0.02, -0.08, 0.06])
        
        max_dd, start_idx, end_idx = await metrics_calculator.maximum_drawdown(returns)
        
        assert max_dd > 0
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)
        assert start_idx <= end_idx
    
    @pytest.mark.asyncio
    async def test_tracking_error(self, metrics_calculator):
        """Test tracking error calculation"""
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        portfolio_returns = benchmark_returns + np.random.normal(0, 0.005, 252)
        
        te_daily = await metrics_calculator.tracking_error(
            portfolio_returns, benchmark_returns, annualized=False
        )
        te_annual = await metrics_calculator.tracking_error(
            portfolio_returns, benchmark_returns, annualized=True
        )
        
        assert te_daily > 0
        assert te_annual > 0
        assert te_annual > te_daily
    
    @pytest.mark.asyncio
    async def test_information_ratio(self, metrics_calculator):
        """Test information ratio calculation"""
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        # Portfolio with slight outperformance
        portfolio_returns = benchmark_returns + np.random.normal(0.0002, 0.005, 252)
        
        info_ratio = await metrics_calculator.information_ratio(
            portfolio_returns, benchmark_returns
        )
        
        assert isinstance(info_ratio, float)
        # Should be positive given our construction
        assert info_ratio > 0

class TestRiskCalculator:
    """Test comprehensive risk calculator"""
    
    @pytest.fixture
    def risk_calculator(self):
        return RiskCalculator()
    
    @pytest.fixture
    def sample_multi_asset_data(self):
        """Generate sample multi-asset data"""
        np.random.seed(42)
        n_obs = 252
        
        returns_data = {
            'AAPL': np.random.normal(0.001, 0.02, n_obs),
            'GOOGL': np.random.normal(0.0008, 0.018, n_obs),
            'MSFT': np.random.normal(0.0012, 0.016, n_obs)
        }
        
        return returns_data
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_analysis(self, risk_calculator, sample_multi_asset_data):
        """Test comprehensive risk analysis"""
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        
        results = await risk_calculator.comprehensive_risk_analysis(
            sample_multi_asset_data,
            portfolio_weights={'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3},
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 7],
            benchmark_returns=benchmark_returns
        )
        
        # Check structure
        assert 'var_metrics' in results
        assert 'correlation_analysis' in results
        assert 'portfolio_metrics' in results
        assert 'benchmark_analysis' in results
        
        # Check VaR metrics
        assert 'var_1d_95_historical' in results['var_metrics']
        assert 'var_7d_99_parametric' in results['var_metrics']
        assert 'es_1d_95' in results['var_metrics']
        
        # Check correlation analysis
        assert 'matrix' in results['correlation_analysis']
        assert 'symbols' in results['correlation_analysis']
        assert len(results['correlation_analysis']['symbols']) == 3
        
        # Check portfolio metrics
        assert 'volatility' in results['portfolio_metrics']
        assert 'sharpe_ratio' in results['portfolio_metrics']
        assert 'max_drawdown' in results['portfolio_metrics']
        
        # Check benchmark analysis
        assert 'beta' in results['benchmark_analysis']
        assert 'alpha' in results['benchmark_analysis']
        assert 'tracking_error' in results['benchmark_analysis']
    
    @pytest.mark.asyncio
    async def test_portfolio_returns_calculation(self, risk_calculator, sample_multi_asset_data):
        """Test portfolio returns calculation"""
        weights = {'AAPL': 0.5, 'GOOGL': 0.3, 'MSFT': 0.2}
        
        portfolio_returns = await risk_calculator._calculate_portfolio_returns(
            sample_multi_asset_data, weights
        )
        
        assert len(portfolio_returns) == 252
        assert np.isfinite(portfolio_returns).all()
    
    @pytest.mark.asyncio
    async def test_equal_weight_portfolio(self, risk_calculator, sample_multi_asset_data):
        """Test equal weight portfolio when no weights provided"""
        portfolio_returns = await risk_calculator._calculate_portfolio_returns(
            sample_multi_asset_data, weights=None
        )
        
        assert len(portfolio_returns) == 252
        assert np.isfinite(portfolio_returns).all()

# Error handling tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def var_calculator(self):
        return VaRCalculator()
    
    @pytest.mark.asyncio
    async def test_empty_returns_array(self, var_calculator):
        """Test handling of empty returns array"""
        with pytest.raises(ValueError):
            await var_calculator.historical_var(np.array([]), 0.95, 1)
    
    @pytest.mark.asyncio
    async def test_all_nan_returns(self, var_calculator):
        """Test handling of all NaN returns"""
        returns_all_nan = np.array([np.nan] * 100)
        
        with pytest.raises(ValueError):
            await var_calculator.historical_var(returns_all_nan, 0.95, 1)
    
    @pytest.mark.asyncio
    async def test_invalid_confidence_level(self, var_calculator):
        """Test handling of invalid confidence levels"""
        sample_returns = np.random.normal(0, 0.02, 100)
        
        # Confidence level outside [0, 1] range should raise error
        with pytest.raises(ValueError):
            await var_calculator.historical_var(sample_returns, 1.5, 1)
    
    @pytest.mark.asyncio
    async def test_single_asset_correlation(self):
        """Test correlation matrix with single asset"""
        corr_calculator = CorrelationCalculator()
        single_asset_data = {'AAPL': np.random.normal(0, 0.02, 100)}
        
        with pytest.raises(ValueError):
            await corr_calculator.correlation_matrix(single_asset_data)

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])
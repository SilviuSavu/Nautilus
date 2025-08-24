#!/usr/bin/env python3
"""
Unit Tests for PyFolio Integration
=================================

Comprehensive test suite for the PyFolio analytics integration layer.
Tests performance metrics calculation, tear sheet generation, and error handling.

Test Categories:
- Unit tests for PyFolioAnalytics class
- Performance and timing validation
- Error handling and edge cases
- Cache functionality testing
- HTML and JSON output validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyfolio_integration import (
        PyFolioAnalytics, 
        PyFolioMetrics, 
        TearSheetConfig, 
        AnalyticsStatus,
        quick_analytics
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Import error: {e}")


class TestPyFolioIntegration:
    """Test suite for PyFolio integration functionality"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        # Generate realistic daily returns: ~8% annual return, 15% volatility
        returns = np.random.normal(0.0003, 0.015, len(dates))
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def benchmark_returns(self, sample_returns):
        """Generate benchmark return data"""
        # Slightly lower returns and volatility than portfolio
        np.random.seed(123)
        benchmark = np.random.normal(0.0002, 0.012, len(sample_returns))
        return pd.Series(benchmark, index=sample_returns.index)
    
    @pytest.fixture
    def pyfolio_analytics(self):
        """Create PyFolioAnalytics instance"""
        return PyFolioAnalytics(cache_ttl_minutes=5)
    
    @pytest.fixture
    def tear_sheet_config(self):
        """Create test configuration"""
        return TearSheetConfig(
            risk_free_rate=0.02,
            confidence_level=0.05,
            rolling_window=63  # 3 months for testing
        )
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    def test_pyfolio_initialization(self, pyfolio_analytics):
        """Test PyFolio analytics initialization"""
        assert pyfolio_analytics is not None
        assert isinstance(pyfolio_analytics.cache_ttl, timedelta)
        assert pyfolio_analytics.calculations_performed == 0
        assert pyfolio_analytics.total_calculation_time == 0.0
        assert pyfolio_analytics.cache_hits == 0
        assert pyfolio_analytics.cache_misses == 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, pyfolio_analytics, sample_returns):
        """Test basic performance metrics calculation"""
        portfolio_id = "test_portfolio_001"
        
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns
        )
        
        # Verify metrics object structure
        assert isinstance(metrics, PyFolioMetrics)
        assert metrics.portfolio_id == portfolio_id
        assert metrics.status == AnalyticsStatus.COMPLETED
        assert isinstance(metrics.computation_date, datetime)
        
        # Verify all core metrics are present and valid
        assert isinstance(metrics.total_return, (int, float))
        assert isinstance(metrics.annual_return, (int, float))
        assert isinstance(metrics.annual_volatility, (int, float))
        assert isinstance(metrics.sharpe_ratio, (int, float))
        assert isinstance(metrics.max_drawdown, (int, float))
        
        # Check that we don't have NaN values for core metrics
        core_metrics = [
            metrics.total_return, metrics.annual_return, 
            metrics.annual_volatility, metrics.sharpe_ratio, 
            metrics.max_drawdown
        ]
        
        for metric in core_metrics:
            assert not np.isnan(metric), f"Core metric should not be NaN: {metric}"
        
        # Verify reasonable value ranges
        assert -1.0 < metrics.total_return < 5.0  # Total return between -100% and 500%
        assert -1.0 < metrics.annual_return < 2.0  # Annual return between -100% and 200%
        assert 0 < metrics.annual_volatility < 2.0  # Volatility between 0% and 200%
        assert -10 < metrics.sharpe_ratio < 10  # Reasonable Sharpe ratio range
        assert -1.0 <= metrics.max_drawdown <= 0  # Max drawdown should be negative or zero
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_benchmark_comparison_metrics(self, pyfolio_analytics, sample_returns, benchmark_returns):
        """Test performance metrics calculation with benchmark"""
        portfolio_id = "test_portfolio_benchmark"
        
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns,
            benchmark_returns=benchmark_returns
        )
        
        # Verify benchmark-specific metrics are calculated
        assert metrics.alpha is not None
        assert metrics.beta is not None
        assert metrics.tracking_error is not None
        assert metrics.information_ratio is not None
        
        # Verify reasonable ranges for benchmark metrics
        assert isinstance(metrics.alpha, (int, float))
        assert isinstance(metrics.beta, (int, float))
        assert not np.isnan(metrics.alpha)
        assert not np.isnan(metrics.beta)
        
        # Beta should be reasonable (typically between 0.5 and 2.0)
        assert 0 < metrics.beta < 3.0
        
        # Tracking error should be positive
        assert metrics.tracking_error >= 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_performance_timing_requirements(self, pyfolio_analytics, sample_returns):
        """Test that performance calculations meet <200ms timing requirement"""
        portfolio_id = "test_performance_timing"
        
        start_time = datetime.now()
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns
        )
        end_time = datetime.now()
        
        # Calculate total time
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should meet the 200ms requirement
        assert total_time_ms < 200, f"Calculation took {total_time_ms:.1f}ms, exceeds 200ms requirement"
        
        # Verify internal timing matches
        assert metrics.calculation_time_ms < 200
        assert metrics.calculation_time_ms > 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_error_handling_empty_returns(self, pyfolio_analytics):
        """Test error handling for empty returns data"""
        portfolio_id = "test_empty_returns"
        empty_returns = pd.Series([], dtype=float)
        
        with pytest.raises(ValueError, match="Returns series cannot be empty"):
            await pyfolio_analytics.compute_performance_metrics(
                portfolio_id=portfolio_id,
                returns=empty_returns
            )
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_error_handling_insufficient_data(self, pyfolio_analytics):
        """Test error handling for insufficient data"""
        portfolio_id = "test_insufficient_data"
        insufficient_returns = pd.Series([0.01, 0.005, -0.002])  # Only 3 days
        
        with pytest.raises(ValueError, match="Insufficient data"):
            await pyfolio_analytics.compute_performance_metrics(
                portfolio_id=portfolio_id,
                returns=insufficient_returns
            )
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_cache_functionality(self, pyfolio_analytics, sample_returns):
        """Test caching functionality for repeated calculations"""
        portfolio_id = "test_cache_functionality"
        
        # First calculation - should be cache miss
        initial_cache_misses = pyfolio_analytics.cache_misses
        metrics1 = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns
        )
        assert pyfolio_analytics.cache_misses == initial_cache_misses + 1
        
        # Second calculation with same data - should be cache hit
        initial_cache_hits = pyfolio_analytics.cache_hits
        metrics2 = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns
        )
        assert pyfolio_analytics.cache_hits == initial_cache_hits + 1
        
        # Results should be identical
        assert metrics1.total_return == metrics2.total_return
        assert metrics1.sharpe_ratio == metrics2.sharpe_ratio
        assert metrics1.calculation_time_ms == metrics2.calculation_time_ms
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_tear_sheet_data_generation(self, pyfolio_analytics, sample_returns, tear_sheet_config):
        """Test tear sheet data generation"""
        portfolio_id = "test_tear_sheet_data"
        
        tear_sheet_data = await pyfolio_analytics.generate_tear_sheet_data(
            portfolio_id=portfolio_id,
            returns=sample_returns,
            config=tear_sheet_config
        )
        
        # Verify structure
        assert isinstance(tear_sheet_data, dict)
        assert 'metadata' in tear_sheet_data
        assert 'performance_metrics' in tear_sheet_data
        assert 'rolling_metrics' in tear_sheet_data
        assert 'drawdown_analysis' in tear_sheet_data
        assert 'returns_analysis' in tear_sheet_data
        assert 'risk_analysis' in tear_sheet_data
        
        # Verify metadata
        metadata = tear_sheet_data['metadata']
        assert metadata['portfolio_id'] == portfolio_id
        assert 'generation_time' in metadata
        assert 'total_observations' in metadata
        assert metadata['total_observations'] == len(sample_returns)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_html_tear_sheet_generation(self, pyfolio_analytics, sample_returns):
        """Test HTML tear sheet generation"""
        portfolio_id = "test_html_tear_sheet"
        
        html_output = await pyfolio_analytics.generate_html_tear_sheet(
            portfolio_id=portfolio_id,
            returns=sample_returns
        )
        
        # Verify HTML structure
        assert isinstance(html_output, str)
        assert len(html_output) > 1000  # Should be substantial HTML
        assert '<!DOCTYPE html>' in html_output or '<html>' in html_output
        assert portfolio_id in html_output
        assert 'Portfolio Performance Analysis' in html_output
        assert 'Sharpe Ratio' in html_output
        assert 'Total Return' in html_output
        assert 'Maximum Drawdown' in html_output
        
        # Verify CSS styling is included
        assert '<style>' in html_output
        assert 'font-family' in html_output
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_health_check(self, pyfolio_analytics):
        """Test health check functionality"""
        health_status = await pyfolio_analytics.health_check()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'pyfolio_available' in health_status
        assert 'version' in health_status
        assert 'last_check' in health_status
        
        # If PyFolio is available, should have functionality test
        if health_status['pyfolio_available']:
            assert 'functionality_test' in health_status
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    def test_performance_stats(self, pyfolio_analytics):
        """Test performance statistics gathering"""
        stats = pyfolio_analytics.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'pyfolio_available' in stats
        assert 'calculations_performed' in stats
        assert 'average_calculation_time_ms' in stats
        assert 'cache_statistics' in stats
        assert 'performance_metrics' in stats
        
        # Verify cache statistics structure
        cache_stats = stats['cache_statistics']
        assert 'cache_hits' in cache_stats
        assert 'cache_misses' in cache_stats
        assert 'cache_hit_rate' in cache_stats
        assert 'cached_entries' in cache_stats
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_configuration_handling(self, pyfolio_analytics, sample_returns):
        """Test different configuration options"""
        portfolio_id = "test_configuration"
        
        # Test with custom risk-free rate
        custom_config = TearSheetConfig(
            risk_free_rate=0.05,  # 5% risk-free rate
            confidence_level=0.01,  # 1% VaR
            rolling_window=21  # 1 month rolling window
        )
        
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=sample_returns,
            config=custom_config
        )
        
        # Should complete successfully with custom config
        assert metrics.status == AnalyticsStatus.COMPLETED
        assert isinstance(metrics.sharpe_ratio, (int, float))
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, pyfolio_analytics):
        """Test performance with larger dataset"""
        portfolio_id = "test_large_dataset"
        
        # Generate 5 years of daily data
        np.random.seed(42)
        dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
        
        start_time = datetime.now()
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=returns
        )
        end_time = datetime.now()
        
        calculation_time = (end_time - start_time).total_seconds() * 1000
        
        # Should still complete within reasonable time even with large dataset
        assert calculation_time < 500  # 500ms for large dataset
        assert metrics.status == AnalyticsStatus.COMPLETED
        assert len(returns) > 1800  # Verify we have substantial data
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_missing_pyfolio_handling(self):
        """Test behavior when PyFolio is not available"""
        with patch('pyfolio_integration.PYFOLIO_AVAILABLE', False):
            analytics = PyFolioAnalytics()
            assert not analytics.available
            
            # Should raise RuntimeError when trying to compute metrics
            with pytest.raises(RuntimeError, match="PyFolio not available"):
                await analytics.compute_performance_metrics(
                    "test_portfolio",
                    pd.Series([0.01, 0.005, -0.002, 0.003])
                )
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, pyfolio_analytics):
        """Test with extreme market conditions"""
        portfolio_id = "test_extreme_conditions"
        
        # Create extreme returns: crash followed by recovery
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        extreme_returns = []
        
        # Normal period
        extreme_returns.extend(np.random.normal(0.001, 0.01, 50))
        # Crash period
        extreme_returns.extend([-0.20, -0.15, -0.10, -0.08, -0.05])
        # Recovery period  
        extreme_returns.extend(np.random.normal(0.05, 0.03, 45))
        
        returns = pd.Series(extreme_returns, index=dates)
        
        metrics = await pyfolio_analytics.compute_performance_metrics(
            portfolio_id=portfolio_id,
            returns=returns
        )
        
        # Should handle extreme conditions gracefully
        assert metrics.status == AnalyticsStatus.COMPLETED
        assert metrics.max_drawdown < -0.1  # Should capture the crash
        assert not np.isnan(metrics.sharpe_ratio)  # Should still compute Sharpe ratio


class TestQuickAnalyticsFunction:
    """Test the standalone quick_analytics function"""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_quick_analytics_basic(self):
        """Test basic quick analytics functionality"""
        portfolio_id = "quick_test_001"
        returns_data = [0.01, -0.005, 0.02, 0.005, -0.01, 0.015, -0.008] * 10  # 70 days
        
        result = await quick_analytics(portfolio_id, returns_data)
        
        assert isinstance(result, dict)
        assert 'portfolio_id' in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert result['portfolio_id'] == portfolio_id
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_quick_analytics_with_benchmark(self):
        """Test quick analytics with benchmark"""
        portfolio_id = "quick_test_benchmark"
        returns_data = [0.01, -0.005, 0.02, 0.005, -0.01, 0.015, -0.008] * 10
        benchmark_data = [0.008, -0.002, 0.015, 0.003, -0.012, 0.01, -0.005] * 10
        
        result = await quick_analytics(portfolio_id, returns_data, benchmark_data)
        
        assert isinstance(result, dict)
        assert 'alpha' in result
        assert 'beta' in result
        assert isinstance(result['alpha'], (int, float))
        assert isinstance(result['beta'], (int, float))
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PyFolio integration not available")
    @pytest.mark.asyncio
    async def test_quick_analytics_error_handling(self):
        """Test quick analytics error handling"""
        portfolio_id = "quick_test_error"
        invalid_returns = []  # Empty returns
        
        result = await quick_analytics(portfolio_id, invalid_returns)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'status' in result
        assert result['status'] == 'failed'


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
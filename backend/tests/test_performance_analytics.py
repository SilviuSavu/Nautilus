"""
STORY 5.1 PERFORMANCE ANALYTICS - COMPREHENSIVE TESTING
Tests all advanced performance analytics backend API endpoints

This test suite validates complete performance analytics functionality.
"""

import pytest
import asyncio
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import numpy as np
from datetime import datetime, timedelta

# Import the FastAPI app and services
from main import app
from portfolio_service import portfolio_service
from performance_analytics_routes import MonteCarloRequest

client = TestClient(app)


class TestPerformanceAnalyticsAPIs:
    """Test suite for Story 5.1 Performance Analytics backend APIs"""
    
    def setup_method(self):
        """Setup test data for each test"""
        
        # Mock portfolio positions for testing
        self.mock_positions = [
            Mock(
                instrument_id="AAPL",
                venue=Mock(value="SMART"),
                quantity=100,
                entry_price=145.0,
                current_price=150.0,
                market_value=15000.0,
                unrealized_pnl=500.0,
                symbol="AAPL"
            ),
            Mock(
                instrument_id="GOOGL", 
                venue=Mock(value="SMART"),
                quantity=25,
                entry_price=2800.0,
                current_price=2750.0,
                market_value=68750.0,
                unrealized_pnl=-1250.0,
                symbol="GOOGL"
            ),
            Mock(
                instrument_id="TSLA",
                venue=Mock(value="SMART"), 
                quantity=50,
                entry_price=220.0,
                current_price=210.0,
                market_value=10500.0,
                unrealized_pnl=-500.0,
                symbol="TSLA"
            )
        ]

    @patch.object(portfolio_service, 'get_positions')
    def test_performance_analytics_endpoint(self, mock_get_positions):
        """Test GET /api/v1/analytics/performance/{portfolio_id}"""
        
        mock_get_positions.return_value = self.mock_positions
        
        response = client.get("/api/v1/analytics/performance/test_portfolio?benchmark=SPY")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'alpha', 'beta', 'information_ratio', 'tracking_error', 
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
            'volatility', 'downside_deviation', 'rolling_metrics',
            'period_start', 'period_end', 'benchmark'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify data types and ranges
        assert isinstance(data['alpha'], float)
        assert isinstance(data['beta'], float) 
        assert isinstance(data['sharpe_ratio'], float)
        assert isinstance(data['rolling_metrics'], list)
        assert data['benchmark'] == 'SPY'
        
        print("✅ Performance analytics endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_monte_carlo_simulation_endpoint(self, mock_get_positions):
        """Test POST /api/v1/analytics/monte-carlo"""
        
        mock_get_positions.return_value = self.mock_positions
        
        request_data = {
            "portfolio_id": "test_portfolio",
            "scenarios": 1000,
            "time_horizon_days": 30,
            "confidence_levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "stress_scenarios": ["market_crash", "high_volatility"]
        }
        
        response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'scenarios_run', 'time_horizon_days', 'confidence_intervals',
            'expected_return', 'probability_of_loss', 'value_at_risk_5',
            'expected_shortfall_5', 'worst_case_scenario', 'best_case_scenario',
            'stress_test_results', 'simulation_paths'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify simulation results
        assert data['scenarios_run'] == 1000
        assert data['time_horizon_days'] == 30
        assert len(data['stress_test_results']) == 2  # Two stress scenarios
        assert isinstance(data['simulation_paths'], list)
        
        # Verify confidence intervals
        confidence = data['confidence_intervals']
        assert 'percentile_5' in confidence
        assert 'percentile_50' in confidence
        assert 'percentile_95' in confidence
        
        # Verify ordering of percentiles
        assert confidence['percentile_5'] <= confidence['percentile_50'] <= confidence['percentile_95']
        
        print("✅ Monte Carlo simulation endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_attribution_analysis_endpoint(self, mock_get_positions):
        """Test GET /api/v1/analytics/attribution/{portfolio_id}"""
        
        mock_get_positions.return_value = self.mock_positions
        
        response = client.get("/api/v1/analytics/attribution/test_portfolio?attribution_type=sector&period=3M")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'attribution_type', 'period_start', 'period_end', 'total_active_return',
            'attribution_breakdown', 'sector_attribution', 'factor_attribution'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify attribution breakdown
        breakdown = data['attribution_breakdown']
        assert 'security_selection' in breakdown
        assert 'asset_allocation' in breakdown
        assert 'interaction_effect' in breakdown
        
        # Verify sector attribution structure
        if data['sector_attribution']:
            sector = data['sector_attribution'][0]
            sector_fields = [
                'sector', 'portfolio_weight', 'benchmark_weight',
                'portfolio_return', 'benchmark_return', 'allocation_effect',
                'selection_effect', 'total_effect'
            ]
            for field in sector_fields:
                assert field in sector, f"Missing sector field: {field}"
        
        # Verify factor attribution structure
        if data['factor_attribution']:
            factor = data['factor_attribution'][0]
            factor_fields = ['factor_name', 'factor_exposure', 'factor_return', 'contribution']
            for field in factor_fields:
                assert field in factor, f"Missing factor field: {field}"
        
        print("✅ Attribution analysis endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_statistical_tests_endpoint(self, mock_get_positions):
        """Test GET /api/v1/analytics/statistical-tests/{portfolio_id}"""
        
        mock_get_positions.return_value = self.mock_positions
        
        response = client.get("/api/v1/analytics/statistical-tests/test_portfolio?test_type=sharpe&significance_level=0.05")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'sharpe_ratio_test', 'alpha_significance_test', 'beta_stability_test',
            'performance_persistence', 'bootstrap_results'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify Sharpe ratio test structure
        sharpe_test = data['sharpe_ratio_test']
        sharpe_fields = ['sharpe_ratio', 't_statistic', 'p_value', 'is_significant', 'confidence_interval']
        for field in sharpe_fields:
            assert field in sharpe_test, f"Missing Sharpe test field: {field}"
        
        # Verify alpha test structure
        alpha_test = data['alpha_significance_test']
        alpha_fields = ['alpha', 't_statistic', 'p_value', 'is_significant', 'confidence_interval']
        for field in alpha_fields:
            assert field in alpha_test, f"Missing alpha test field: {field}"
        
        # Verify beta stability test
        beta_test = data['beta_stability_test']
        beta_fields = ['beta', 'rolling_beta_std', 'stability_score', 'regime_changes_detected']
        for field in beta_fields:
            assert field in beta_test, f"Missing beta test field: {field}"
        
        # Verify performance persistence
        persistence = data['performance_persistence']
        persistence_fields = ['persistence_score', 'consecutive_winning_periods', 'consistency_rating']
        for field in persistence_fields:
            assert field in persistence, f"Missing persistence field: {field}"
        
        # Verify bootstrap results
        assert isinstance(data['bootstrap_results'], list)
        if data['bootstrap_results']:
            bootstrap = data['bootstrap_results'][0]
            bootstrap_fields = ['metric', 'bootstrap_mean', 'bootstrap_std', 'confidence_interval_95']
            for field in bootstrap_fields:
                assert field in bootstrap, f"Missing bootstrap field: {field}"
        
        print("✅ Statistical tests endpoint test passed")

    def test_benchmarks_endpoint(self):
        """Test GET /api/v1/analytics/benchmarks"""
        
        response = client.get("/api/v1/analytics/benchmarks")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert 'benchmarks' in data
        assert isinstance(data['benchmarks'], list)
        assert len(data['benchmarks']) > 0
        
        # Verify benchmark structure
        benchmark = data['benchmarks'][0]
        benchmark_fields = ['symbol', 'name', 'category', 'data_available_from']
        for field in benchmark_fields:
            assert field in benchmark, f"Missing benchmark field: {field}"
        
        # Verify expected benchmarks are present
        symbols = [b['symbol'] for b in data['benchmarks']]
        expected_symbols = ['SPY', 'QQQ', 'IWM', 'VTI']
        for symbol in expected_symbols:
            assert symbol in symbols, f"Missing expected benchmark: {symbol}"
        
        print("✅ Benchmarks endpoint test passed")

    def test_api_error_handling(self):
        """Test API endpoint error handling for invalid inputs"""
        
        # Test missing portfolio_id
        response = client.get("/api/v1/analytics/performance/")
        assert response.status_code == 404  # Not found for empty portfolio_id
        
        # Test invalid attribution type
        response = client.get("/api/v1/analytics/attribution/test_portfolio?attribution_type=invalid&period=3M")
        assert response.status_code == 400
        
        # Test invalid significance level
        response = client.get("/api/v1/analytics/statistical-tests/test_portfolio?significance_level=1.5")
        assert response.status_code == 400
        
        # Test invalid Monte Carlo request
        invalid_request = {
            "portfolio_id": "test_portfolio",
            "scenarios": -100,  # Invalid
            "time_horizon_days": 500  # Invalid
        }
        response = client.post("/api/v1/analytics/monte-carlo", json=invalid_request)
        assert response.status_code == 400
        
        print("✅ Error handling tests passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_empty_portfolio_handling(self, mock_get_positions):
        """Test handling of empty portfolios"""
        
        mock_get_positions.return_value = []  # Empty portfolio
        
        # Test performance analytics with empty portfolio
        response = client.get("/api/v1/analytics/performance/empty_portfolio")
        assert response.status_code == 200
        data = response.json()
        assert data['alpha'] == 0.0
        assert data['beta'] == 1.0
        assert data['sharpe_ratio'] == 0.0
        
        # Test Monte Carlo with empty portfolio
        request_data = {
            "portfolio_id": "empty_portfolio",
            "scenarios": 100,
            "time_horizon_days": 30
        }
        response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data['expected_return'] == 0.0
        assert data['probability_of_loss'] == 0.0
        
        print("✅ Empty portfolio handling test passed")

    def test_api_response_performance(self):
        """Test that API responses are reasonably fast"""
        
        import time
        
        with patch.object(portfolio_service, 'get_positions', return_value=self.mock_positions):
            
            # Test performance analytics response time
            start_time = time.time()
            response = client.get("/api/v1/analytics/performance/test_portfolio")
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < 5.0, f"Performance analytics too slow: {response_time:.2f}s"
            
            # Test Monte Carlo response time (smaller simulation)
            request_data = {
                "portfolio_id": "test_portfolio",
                "scenarios": 100,  # Reduced for performance
                "time_horizon_days": 10
            }
            
            start_time = time.time()
            response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < 10.0, f"Monte Carlo too slow: {response_time:.2f}s"
            
            print(f"✅ API response performance test passed (Analytics: {response_time:.3f}s)")

    def test_monte_carlo_parameter_validation(self):
        """Test Monte Carlo parameter validation and edge cases"""
        
        with patch.object(portfolio_service, 'get_positions', return_value=self.mock_positions):
            
            # Test minimum valid parameters
            request_data = {
                "portfolio_id": "test_portfolio",
                "scenarios": 1,
                "time_horizon_days": 1
            }
            response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
            assert response.status_code == 200
            
            # Test maximum valid parameters
            request_data = {
                "portfolio_id": "test_portfolio", 
                "scenarios": 100000,
                "time_horizon_days": 365
            }
            response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
            assert response.status_code == 200
            
            # Test with stress scenarios
            request_data = {
                "portfolio_id": "test_portfolio",
                "scenarios": 1000,
                "time_horizon_days": 30,
                "stress_scenarios": ["market_crash", "recession", "high_volatility"]
            }
            response = client.post("/api/v1/analytics/monte-carlo", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert len(data['stress_test_results']) == 3
            
            print("✅ Monte Carlo parameter validation test passed")


class TestPerformanceAnalyticsIntegration:
    """Integration tests with portfolio and risk services"""
    
    @patch('performance_analytics_routes.risk_service')
    @patch.object(portfolio_service, 'get_positions')
    def test_risk_service_integration(self, mock_get_positions, mock_risk_service):
        """Test integration with risk service for volatility estimates"""
        
        # Setup mocks
        mock_positions = [
            Mock(
                instrument_id="AAPL",
                venue=Mock(value="SMART"),
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                market_value=15000.0,
                unrealized_pnl=0.0,
                symbol="AAPL"
            )
        ]
        mock_get_positions.return_value = mock_positions
        
        # Mock risk service response
        mock_risk_service.calculate_position_risk = AsyncMock(return_value={
            'risk_metrics': {
                'volatility': 0.25,  # 25% annual volatility
                'var_1d': 500.0
            }
        })
        
        # Test performance analytics with risk integration
        response = client.get("/api/v1/analytics/performance/test_portfolio")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify risk integration worked
        assert data['volatility'] > 0  # Should have calculated volatility
        
        print("✅ Risk service integration test passed")

    def test_benchmark_data_simulation(self):
        """Test benchmark data simulation and comparison logic"""
        
        with patch.object(portfolio_service, 'get_positions') as mock_get_positions:
            mock_get_positions.return_value = [
                Mock(
                    instrument_id="AAPL",
                    venue=Mock(value="SMART"),
                    quantity=100,
                    entry_price=150.0,
                    current_price=155.0,
                    market_value=15500.0,
                    unrealized_pnl=500.0,
                    symbol="AAPL"
                )
            ]
            
            # Test with different benchmarks
            benchmarks = ["SPY", "QQQ", "IWM"]
            
            for benchmark in benchmarks:
                response = client.get(f"/api/v1/analytics/performance/test_portfolio?benchmark={benchmark}")
                assert response.status_code == 200
                data = response.json()
                assert data['benchmark'] == benchmark
                assert 'alpha' in data
                assert 'beta' in data
        
        print("✅ Benchmark data simulation test passed")


def run_comprehensive_performance_analytics_tests():
    """Run all Story 5.1 Performance Analytics backend API tests"""
    
    print("🚀 STORY 5.1 PERFORMANCE ANALYTICS - COMPREHENSIVE BACKEND API TESTING")
    print("=" * 90)
    
    # Initialize test classes
    api_tests = TestPerformanceAnalyticsAPIs()
    integration_tests = TestPerformanceAnalyticsIntegration()
    
    try:
        print("\n📊 Testing Performance Analytics API Endpoints...")
        
        # Run API endpoint tests
        api_tests.setup_method()
        api_tests.test_performance_analytics_endpoint()
        
        api_tests.setup_method()
        api_tests.test_monte_carlo_simulation_endpoint()
        
        api_tests.setup_method()
        api_tests.test_attribution_analysis_endpoint()
        
        api_tests.setup_method()
        api_tests.test_statistical_tests_endpoint()
        
        api_tests.test_benchmarks_endpoint()
        
        # Error handling and edge case tests
        api_tests.test_api_error_handling()
        api_tests.setup_method()
        api_tests.test_empty_portfolio_handling()
        api_tests.test_api_response_performance()
        api_tests.test_monte_carlo_parameter_validation()
        
        print("\n🔗 Testing Integration with Data Services...")
        integration_tests.test_risk_service_integration()
        integration_tests.test_benchmark_data_simulation()
        
        print("\n" + "=" * 90)
        print("🎉 ALL STORY 5.1 PERFORMANCE ANALYTICS BACKEND API TESTS PASSED!")
        print("✅ Advanced Performance Analytics backend implementation is complete and tested")
        print("✅ All API endpoints functional with comprehensive error handling")
        print("✅ Monte Carlo simulation engine operational with stress testing")
        print("✅ Statistical significance testing and attribution analysis working")
        print("✅ Integration with portfolio and risk services validated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ STORY 5.1 BACKEND TEST FAILED: {e}")
        print("🚨 Performance Analytics backend needs additional work")
        return False


if __name__ == "__main__":
    success = run_comprehensive_performance_analytics_tests()
    if not success:
        exit(1)
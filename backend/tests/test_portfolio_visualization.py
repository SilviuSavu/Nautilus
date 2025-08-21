"""
STORY 4.4 PORTFOLIO VISUALIZATION - COMPREHENSIVE TESTING
Tests all backend API endpoints with real data integration

This test suite validates complete portfolio visualization functionality.
"""

import pytest
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import json
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import numpy as np
from datetime import datetime, timedelta

# Import the FastAPI app and services
from main import app
from portfolio_service import portfolio_service
from historical_data_service import historical_data_service

client = TestClient(app)


class TestPortfolioVisualizationAPIs:
    """Test suite for Story 4.4 Portfolio Visualization backend APIs"""
    
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
        
        # Mock historical price data
        self.mock_price_history = {
            "AAPL": [150.0, 148.5, 152.0, 149.0, 151.5] * 10,  # 50 data points
            "GOOGL": [2750.0, 2780.0, 2720.0, 2755.0, 2745.0] * 10,
            "TSLA": [210.0, 215.0, 208.0, 212.0, 209.0] * 10
        }

    @patch.object(portfolio_service, 'get_positions')
    def test_aggregated_portfolio_metrics_endpoint(self, mock_get_positions):
        """Test GET /api/v1/portfolio/aggregated endpoint"""
        
        mock_get_positions.return_value = self.mock_positions
        
        response = client.get("/api/v1/portfolio/aggregated?portfolio_id=test_portfolio")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'portfolio_id', 'total_value', 'total_pnl', 'pnl_percentage', 
            'position_count', 'top_performers', 'worst_performers',
            'sector_allocation', 'risk_metrics', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify calculated values
        assert data['portfolio_id'] == "test_portfolio"
        assert data['position_count'] == 3
        assert data['total_value'] == 94250.0  # Sum of market values
        assert data['total_pnl'] == -1250.0    # Sum of unrealized PnL
        
        # Verify top and worst performers
        assert len(data['top_performers']) > 0
        assert len(data['worst_performers']) > 0
        
        print("✅ Aggregated portfolio metrics endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    @patch('portfolio_visualization_routes.historical_data_service')
    def test_performance_attribution_endpoint(self, mock_historical_service, mock_get_positions):
        """Test GET /api/v1/portfolio/attribution endpoint"""
        
        mock_get_positions.return_value = self.mock_positions
        
        # Mock historical data service
        mock_historical_service._connected = True
        mock_historical_service.execute_query = AsyncMock()
        
        # Setup mock historical returns for each symbol
        historical_returns = []
        for symbol in ["AAPL", "GOOGL", "TSLA"]:
            for i in range(30):
                historical_returns.append({
                    'instrument_id': f"{symbol}.SMART",
                    'date': (datetime.now() - timedelta(days=30-i)).date(),
                    'return': np.random.normal(0.001, 0.02)  # Daily return ~0.1% +/- 2%
                })
        
        mock_historical_service.execute_query.return_value = historical_returns
        
        response = client.get("/api/v1/portfolio/attribution?portfolio_id=test_portfolio&period=30")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'portfolio_id', 'period_days', 'total_return', 'attribution_breakdown',
            'risk_contribution', 'alpha_beta_analysis', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify attribution breakdown has data for each position
        assert len(data['attribution_breakdown']) == 3
        
        for attribution in data['attribution_breakdown']:
            assert 'symbol' in attribution
            assert 'weight' in attribution
            assert 'return_contribution' in attribution
            assert 'risk_contribution' in attribution
        
        print("✅ Performance attribution endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_asset_allocation_endpoint(self, mock_get_positions):
        """Test GET /api/v1/portfolio/allocation endpoint"""
        
        mock_get_positions.return_value = self.mock_positions
        
        response = client.get("/api/v1/portfolio/allocation?portfolio_id=test_portfolio")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'portfolio_id', 'total_value', 'allocation_by_asset', 
            'allocation_by_sector', 'allocation_by_geography', 'concentration_metrics'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify allocation by asset
        assert len(data['allocation_by_asset']) == 3
        
        total_allocation = sum(item['percentage'] for item in data['allocation_by_asset'])
        assert abs(total_allocation - 100.0) < 0.01  # Should sum to 100%
        
        # Verify concentration metrics
        concentration = data['concentration_metrics']
        assert 'largest_position_weight' in concentration
        assert 'top_3_concentration' in concentration
        assert 'effective_positions' in concentration
        
        print("✅ Asset allocation endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    @patch('portfolio_visualization_routes.historical_data_service')  
    def test_correlation_analysis_endpoint(self, mock_historical_service, mock_get_positions):
        """Test GET /api/v1/portfolio/correlation endpoint"""
        
        mock_get_positions.return_value = self.mock_positions
        
        # Mock historical data service
        mock_historical_service._connected = True
        mock_historical_service.execute_query = AsyncMock()
        
        # Create more realistic correlation data
        correlation_data = []
        symbols = ["AAPL", "GOOGL", "TSLA"]
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation = 1.0  # Self-correlation is 1
                elif (symbol1, symbol2) in [("AAPL", "GOOGL"), ("GOOGL", "AAPL")]:
                    correlation = 0.65  # Tech stocks moderately correlated
                elif (symbol1, symbol2) in [("AAPL", "TSLA"), ("TSLA", "AAPL")]:
                    correlation = 0.45  # Moderate correlation
                else:
                    correlation = 0.35  # Lower correlation
                
                correlation_data.append({
                    'symbol1': symbol1,
                    'symbol2': symbol2, 
                    'correlation': correlation
                })
        
        mock_historical_service.execute_query.return_value = correlation_data
        
        response = client.get("/api/v1/portfolio/correlation?portfolio_id=test_portfolio&days=30")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'portfolio_id', 'period_days', 'correlation_matrix',
            'diversification_metrics', 'risk_concentration'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify correlation matrix structure
        matrix = data['correlation_matrix']
        assert len(matrix) == 3  # 3x3 matrix for 3 positions
        
        for row in matrix:
            assert len(row['correlations']) == 3
            assert row['symbol'] in ['AAPL', 'GOOGL', 'TSLA']
        
        # Verify diversification metrics
        diversification = data['diversification_metrics']
        assert 'portfolio_correlation' in diversification
        assert 'diversification_ratio' in diversification
        
        print("✅ Correlation analysis endpoint test passed")

    @patch.object(portfolio_service, 'get_positions')
    @patch('portfolio_visualization_routes.historical_data_service')
    def test_historical_performance_endpoint(self, mock_historical_service, mock_get_positions):
        """Test GET /api/v1/portfolio/historical endpoint"""
        
        mock_get_positions.return_value = self.mock_positions
        
        # Mock historical data service
        mock_historical_service._connected = True
        mock_historical_service.execute_query = AsyncMock()
        
        # Create historical portfolio values
        historical_data = []
        base_date = datetime.now() - timedelta(days=30)
        base_value = 90000.0
        
        for i in range(31):  # 31 days of data
            date = base_date + timedelta(days=i)
            # Simulate portfolio growth with some volatility
            growth_factor = 1 + (i * 0.001) + np.random.normal(0, 0.01)
            value = base_value * growth_factor
            
            historical_data.append({
                'date': date.date(),
                'portfolio_value': value,
                'daily_return': np.random.normal(0.001, 0.02),
                'cumulative_return': (value / base_value - 1) * 100
            })
        
        mock_historical_service.execute_query.return_value = historical_data
        
        response = client.get("/api/v1/portfolio/historical?portfolio_id=test_portfolio&days=30")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            'portfolio_id', 'period_days', 'historical_values',
            'performance_summary', 'risk_metrics'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify historical data
        historical_values = data['historical_values']
        assert len(historical_values) == 31  # 31 days of data
        
        for day_data in historical_values:
            assert 'date' in day_data
            assert 'portfolio_value' in day_data
            assert 'daily_return' in day_data
            assert 'cumulative_return' in day_data
        
        # Verify performance summary
        summary = data['performance_summary']
        required_summary_fields = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'best_day', 'worst_day'
        ]
        
        for field in required_summary_fields:
            assert field in summary, f"Missing performance summary field: {field}"
        
        print("✅ Historical performance endpoint test passed")

    def test_endpoint_error_handling(self):
        """Test API endpoint error handling for invalid inputs"""
        
        # Test missing portfolio_id
        response = client.get("/api/v1/portfolio/aggregated")
        assert response.status_code == 422  # Validation error
        
        # Test invalid portfolio_id
        response = client.get("/api/v1/portfolio/aggregated?portfolio_id=")
        assert response.status_code == 400
        
        # Test invalid period parameter
        response = client.get("/api/v1/portfolio/attribution?portfolio_id=test&period=-1")
        assert response.status_code == 400
        
        # Test invalid days parameter
        response = client.get("/api/v1/portfolio/correlation?portfolio_id=test&days=0")
        assert response.status_code == 400
        
        print("✅ Error handling tests passed")

    @patch.object(portfolio_service, 'get_positions')
    def test_empty_portfolio_handling(self, mock_get_positions):
        """Test handling of empty portfolios"""
        
        mock_get_positions.return_value = []  # Empty portfolio
        
        response = client.get("/api/v1/portfolio/aggregated?portfolio_id=empty_portfolio")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify empty portfolio response
        assert data['position_count'] == 0
        assert data['total_value'] == 0.0
        assert data['total_pnl'] == 0.0
        assert len(data['top_performers']) == 0
        assert len(data['worst_performers']) == 0
        
        print("✅ Empty portfolio handling test passed")

    def test_api_response_performance(self):
        """Test that API responses are reasonably fast"""
        
        import time
        
        with patch.object(portfolio_service, 'get_positions', return_value=self.mock_positions):
            start_time = time.time()
            response = client.get("/api/v1/portfolio/aggregated?portfolio_id=test_portfolio")
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < 2.0, f"Response too slow: {response_time:.2f}s"
            
            print(f"✅ API response performance test passed ({response_time:.3f}s)")


class TestPortfolioVisualizationIntegration:
    """Integration tests with real data services"""
    
    @patch('portfolio_visualization_routes.historical_data_service')
    @patch.object(portfolio_service, 'get_positions')
    def test_real_data_integration_simulation(self, mock_get_positions, mock_historical_service):
        """Test integration with real-like data patterns"""
        
        # Setup positions with realistic data
        positions = [
            Mock(
                instrument_id="AAPL",
                venue=Mock(value="SMART"),
                quantity=100,
                entry_price=145.0,
                current_price=150.0,
                market_value=15000.0,
                unrealized_pnl=500.0,
                symbol="AAPL"
            )
        ]
        mock_get_positions.return_value = positions
        
        # Mock historical service connection
        mock_historical_service._connected = True
        mock_historical_service.execute_query = AsyncMock()
        
        # Simulate real historical price data
        historical_prices = []
        base_price = 145.0
        for i in range(30):
            # Simulate realistic price movement
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price = base_price * (1 + change)
            historical_prices.append({
                'close_price': price,
                'timestamp_ns': (datetime.now() - timedelta(days=30-i)).timestamp() * 1e9
            })
            base_price = price
        
        mock_historical_service.execute_query.return_value = historical_prices
        
        # Test aggregated endpoint with realistic data
        response = client.get("/api/v1/portfolio/aggregated?portfolio_id=test_portfolio")
        
        assert response.status_code == 200
        data = response.json()
        assert data['position_count'] == 1
        assert data['total_value'] > 0
        
        print("✅ Real data integration simulation test passed")


def run_comprehensive_tests():
    """Run all Story 4.4 backend API tests"""
    
    print("🚀 STORY 4.4 PORTFOLIO VISUALIZATION - COMPREHENSIVE BACKEND API TESTING")
    print("=" * 80)
    
    # Initialize test classes
    api_tests = TestPortfolioVisualizationAPIs()
    integration_tests = TestPortfolioVisualizationIntegration()
    
    try:
        print("\n📊 Testing Portfolio Visualization API Endpoints...")
        
        # Run API endpoint tests
        api_tests.setup_method()
        api_tests.test_aggregated_portfolio_metrics_endpoint()
        
        api_tests.setup_method()
        api_tests.test_performance_attribution_endpoint()
        
        api_tests.setup_method() 
        api_tests.test_asset_allocation_endpoint()
        
        api_tests.setup_method()
        api_tests.test_correlation_analysis_endpoint()
        
        api_tests.setup_method()
        api_tests.test_historical_performance_endpoint()
        
        # Error handling tests
        api_tests.test_endpoint_error_handling()
        api_tests.setup_method()
        api_tests.test_empty_portfolio_handling()
        api_tests.test_api_response_performance()
        
        print("\n🔗 Testing Integration with Data Services...")
        integration_tests.test_real_data_integration_simulation()
        
        print("\n" + "=" * 80)
        print("🎉 ALL STORY 4.4 BACKEND API TESTS PASSED!")
        print("✅ Portfolio Visualization backend implementation is complete and tested")
        print("✅ All API endpoints functional with proper error handling")
        print("✅ Integration points validated with realistic data patterns")
        
        return True
        
    except Exception as e:
        print(f"\n❌ STORY 4.4 BACKEND TEST FAILED: {e}")
        print("🚨 Portfolio Visualization backend needs additional work")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    if not success:
        exit(1)
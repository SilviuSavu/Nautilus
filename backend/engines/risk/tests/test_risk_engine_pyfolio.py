#!/usr/bin/env python3
"""
Integration Tests for Risk Engine PyFolio Endpoints
==================================================

Tests the PyFolio integration at the FastAPI endpoint level.
Validates API contracts, response formats, and end-to-end functionality.

Test Categories:
- API endpoint functionality
- Request/response validation
- Performance requirements
- Error handling and edge cases
- HTML and JSON output validation
"""

import pytest
import json
import time
from datetime import datetime
from fastapi.testclient import TestClient
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from risk_engine import RiskEngine
    RISK_ENGINE_AVAILABLE = True
except ImportError as e:
    RISK_ENGINE_AVAILABLE = False
    print(f"Risk engine import failed: {e}")


class TestRiskEnginePyFolioEndpoints:
    """Integration tests for PyFolio endpoints in Risk Engine"""
    
    @pytest.fixture(scope="class")
    def risk_engine(self):
        """Create Risk Engine instance for testing"""
        if not RISK_ENGINE_AVAILABLE:
            pytest.skip("Risk engine not available")
        return RiskEngine()
    
    @pytest.fixture(scope="class")
    def client(self, risk_engine):
        """Create test client"""
        return TestClient(risk_engine.app)
    
    @pytest.fixture
    def sample_analytics_request(self):
        """Sample request data for PyFolio analytics"""
        return {
            "returns": [
                0.01, -0.005, 0.02, 0.005, -0.01, 0.015, -0.008, 0.012,
                -0.003, 0.007, 0.018, -0.009, 0.004, -0.012, 0.025, 0.002,
                -0.006, 0.019, 0.003, -0.008, 0.014, 0.001, -0.011, 0.016,
                0.007, -0.004, 0.022, -0.001, 0.009, -0.013, 0.017, 0.006
            ] * 3,  # 96 days of data
            "benchmark_returns": [
                0.008, -0.002, 0.015, 0.003, -0.012, 0.01, -0.005, 0.009,
                -0.001, 0.005, 0.013, -0.007, 0.002, -0.009, 0.018, 0.001,
                -0.004, 0.014, 0.001, -0.006, 0.011, 0.0, -0.008, 0.012,
                0.005, -0.002, 0.016, 0.0, 0.007, -0.01, 0.013, 0.004
            ] * 3,
            "risk_free_rate": 0.02
        }
    
    @pytest.fixture
    def sample_tear_sheet_request(self):
        """Sample request for tear sheet generation"""
        return {
            "returns": [
                0.012, -0.008, 0.025, 0.003, -0.015, 0.018, -0.004, 0.009,
                -0.002, 0.014, 0.021, -0.007, 0.005, -0.011, 0.019, 0.001,
                -0.006, 0.016, 0.004, -0.009, 0.013, 0.002, -0.008, 0.017,
                0.006, -0.003, 0.020, -0.001, 0.008, -0.012, 0.015, 0.007
            ] * 5,  # 160 days
            "format": "json",
            "config": {
                "risk_free_rate": 0.025,
                "confidence_level": 0.05
            }
        }
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_health_endpoint_includes_pyfolio(self, client):
        """Test that health endpoint includes PyFolio status"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify PyFolio integration status is included
        assert "pyfolio_integration" in data
        pyfolio_status = data["pyfolio_integration"]
        
        assert "available" in pyfolio_status
        assert "version" in pyfolio_status
        assert "calculations_performed" in pyfolio_status
        assert "average_response_time_ms" in pyfolio_status
        assert "meets_performance_target" in pyfolio_status
        
        # If PyFolio is available, check additional fields
        if pyfolio_status["available"]:
            assert isinstance(pyfolio_status["calculations_performed"], int)
            assert isinstance(pyfolio_status["average_response_time_ms"], (int, float))
            assert isinstance(pyfolio_status["meets_performance_target"], bool)
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_pyfolio_analytics_endpoint_success(self, client, sample_analytics_request):
        """Test successful PyFolio analytics computation"""
        portfolio_id = "test_portfolio_001"
        
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json=sample_analytics_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["status"] == "success"
        assert data["portfolio_id"] == portfolio_id
        assert "analytics" in data
        assert "computation_time_ms" in data
        
        # Verify analytics data structure
        analytics = data["analytics"]
        required_fields = [
            "total_return", "annual_return", "annual_volatility",
            "sharpe_ratio", "max_drawdown", "sortino_ratio",
            "calmar_ratio", "value_at_risk_5", "conditional_var_5"
        ]
        
        for field in required_fields:
            assert field in analytics
            assert isinstance(analytics[field], (int, float))
        
        # Verify benchmark metrics are included
        benchmark_fields = ["alpha", "beta", "tracking_error", "information_ratio"]
        for field in benchmark_fields:
            assert field in analytics
            # These can be None or numeric
            if analytics[field] is not None:
                assert isinstance(analytics[field], (int, float))
        
        # Verify performance requirement
        assert data["computation_time_ms"] < 200
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_pyfolio_analytics_endpoint_validation(self, client):
        """Test input validation for analytics endpoint"""
        portfolio_id = "test_validation"
        
        # Test empty returns
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json={"returns": []}
        )
        assert response.status_code == 400
        assert "Returns data is required" in response.json()["detail"]
        
        # Test insufficient data
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json={"returns": [0.01, 0.005, -0.002]}  # Only 3 days
        )
        assert response.status_code == 400
        assert "Minimum 30 days" in response.json()["detail"]
        
        # Test missing returns key
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json={"benchmark_returns": [0.01, 0.005]}
        )
        assert response.status_code == 400
        assert "Returns data is required" in response.json()["detail"]
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_tear_sheet_html_endpoint(self, client):
        """Test HTML tear sheet generation endpoint (GET)"""
        portfolio_id = "test_html_tearsheet"
        
        # Test with sample data (the endpoint generates sample data if none provided)
        response = client.get(f"/risk/analytics/pyfolio/tear-sheet/{portfolio_id}?format=html")
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            # If successful, should return HTML
            assert response.headers["content-type"] == "text/html; charset=utf-8"
            html_content = response.content.decode()
            assert "<html>" in html_content or "<!DOCTYPE html>" in html_content
            assert portfolio_id in html_content or "DEMO" in html_content
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_tear_sheet_json_endpoint(self, client):
        """Test JSON tear sheet generation endpoint (GET)"""
        portfolio_id = "test_json_tearsheet"
        
        response = client.get(f"/risk/analytics/pyfolio/tear-sheet/{portfolio_id}?format=json")
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["format"] == "json"
            assert data["portfolio_id"] == portfolio_id
            assert "tear_sheet_data" in data
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_tear_sheet_post_endpoint(self, client, sample_tear_sheet_request):
        """Test tear sheet generation with POST data"""
        portfolio_id = "test_post_tearsheet"
        
        response = client.post(
            f"/risk/analytics/pyfolio/tear-sheet/{portfolio_id}",
            json=sample_tear_sheet_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["format"] == "json"
        assert data["portfolio_id"] == portfolio_id
        assert "tear_sheet_data" in data
        
        # Verify tear sheet data structure
        tear_sheet = data["tear_sheet_data"]
        assert "metadata" in tear_sheet
        assert "performance_metrics" in tear_sheet
        assert "rolling_metrics" in tear_sheet
        assert "drawdown_analysis" in tear_sheet
        assert "returns_analysis" in tear_sheet
        assert "risk_analysis" in tear_sheet
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_tear_sheet_html_post_endpoint(self, client, sample_tear_sheet_request):
        """Test HTML tear sheet generation with POST data"""
        portfolio_id = "test_html_post_tearsheet"
        
        # Modify request for HTML format
        html_request = sample_tear_sheet_request.copy()
        html_request["format"] = "html"
        
        response = client.post(
            f"/risk/analytics/pyfolio/tear-sheet/{portfolio_id}",
            json=html_request
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        
        html_content = response.content.decode()
        assert "<html>" in html_content or "<!DOCTYPE html>" in html_content
        assert portfolio_id in html_content
        assert "Portfolio Performance Analysis" in html_content
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_pyfolio_health_endpoint(self, client):
        """Test PyFolio health check endpoint"""
        response = client.get("/risk/analytics/pyfolio/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify health check structure
        assert "status" in data
        assert "pyfolio_available" in data
        assert "version" in data
        assert "last_check" in data
        
        # If PyFolio is available, check for functionality test
        if data["pyfolio_available"]:
            assert "functionality_test" in data
            if data["functionality_test"] == "passed":
                assert "test_calculation_time_ms" in data
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_pyfolio_performance_endpoint(self, client):
        """Test PyFolio performance statistics endpoint"""
        response = client.get("/risk/analytics/pyfolio/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "performance_statistics" in data
        
        stats = data["performance_statistics"]
        assert "pyfolio_available" in stats
        assert "calculations_performed" in stats
        assert "cache_statistics" in stats
        assert "performance_metrics" in stats
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_demo_tear_sheet_endpoint(self, client):
        """Test demo tear sheet generation endpoint"""
        portfolio_id = "demo_portfolio"
        
        response = client.get(f"/risk/analytics/pyfolio/demo/{portfolio_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        
        html_content = response.content.decode()
        assert "<html>" in html_content or "<!DOCTYPE html>" in html_content
        assert f"DEMO_{portfolio_id}" in html_content
        assert "Portfolio Performance Analysis" in html_content
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_endpoint_performance_requirements(self, client, sample_analytics_request):
        """Test that all endpoints meet performance requirements"""
        portfolio_id = "test_performance_req"
        
        # Test analytics endpoint performance
        start_time = time.time()
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json=sample_analytics_request
        )
        end_time = time.time()
        
        assert response.status_code == 200
        elapsed_ms = (end_time - start_time) * 1000
        
        # Should complete within performance requirement
        assert elapsed_ms < 500  # 500ms for full HTTP round trip
        
        # Check internal computation time
        data = response.json()
        assert data["computation_time_ms"] < 200  # Core requirement
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_concurrent_requests(self, client, sample_analytics_request):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading
        
        portfolio_base = "concurrent_test"
        
        def make_request(portfolio_num):
            portfolio_id = f"{portfolio_base}_{portfolio_num}"
            response = client.post(
                f"/risk/analytics/pyfolio/{portfolio_id}",
                json=sample_analytics_request
            )
            return response.status_code, portfolio_id
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, portfolio_id in results:
            assert status_code == 200
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_error_response_format(self, client):
        """Test that error responses are properly formatted"""
        portfolio_id = "test_error_format"
        
        # Send invalid data to trigger error
        invalid_request = {"returns": "invalid_data"}
        
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json=invalid_request
        )
        
        # Should return proper HTTP error
        assert response.status_code in [400, 422, 500]
        
        # Should have error details
        error_data = response.json()
        assert "detail" in error_data
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_large_dataset_handling(self, client):
        """Test handling of large datasets"""
        portfolio_id = "test_large_dataset"
        
        # Generate large dataset (2 years of daily data)
        import random
        random.seed(42)
        large_returns = [random.gauss(0.0005, 0.015) for _ in range(500)]
        
        large_request = {
            "returns": large_returns,
            "risk_free_rate": 0.02
        }
        
        start_time = time.time()
        response = client.post(
            f"/risk/analytics/pyfolio/{portfolio_id}",
            json=large_request
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should still complete within reasonable time
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 1000  # 1 second for large dataset
        
        data = response.json()
        assert data["status"] == "success"
        assert data["computation_time_ms"] < 500  # Internal computation time


class TestRiskEngineIntegration:
    """Test overall integration with the risk engine"""
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_pyfolio_integration_in_risk_engine(self):
        """Test that PyFolio integration is properly initialized in risk engine"""
        risk_engine = RiskEngine()
        
        # Should have PyFolio analytics instance
        assert hasattr(risk_engine, 'pyfolio')
        assert risk_engine.pyfolio is not None
        
        # Should be properly configured
        assert hasattr(risk_engine.pyfolio, 'cache_ttl')
        assert hasattr(risk_engine.pyfolio, 'available')
    
    @pytest.mark.skipif(not RISK_ENGINE_AVAILABLE, reason="Risk engine not available")
    def test_endpoint_registration(self):
        """Test that PyFolio endpoints are properly registered"""
        risk_engine = RiskEngine()
        client = TestClient(risk_engine.app)
        
        # Test that endpoints exist by checking OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check for PyFolio endpoints
        pyfolio_endpoints = [
            "/risk/analytics/pyfolio/{portfolio_id}",
            "/risk/analytics/pyfolio/tear-sheet/{portfolio_id}",
            "/risk/analytics/pyfolio/health",
            "/risk/analytics/pyfolio/performance",
            "/risk/analytics/pyfolio/demo/{portfolio_id}"
        ]
        
        for endpoint in pyfolio_endpoints:
            assert endpoint in paths, f"Endpoint {endpoint} not found in OpenAPI spec"


# Pytest configuration
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
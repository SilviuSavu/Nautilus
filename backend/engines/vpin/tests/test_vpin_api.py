"""
VPIN API Tests
Comprehensive testing for all VPIN Engine REST API endpoints.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# FastAPI testing
from fastapi.testclient import TestClient
from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

# VPIN imports
from backend.engines.vpin.routes import vpin_router
from backend.engines.vpin.vpin_engine import VPINEngine
from backend.engines.vpin.models import (
    VPINSignal, MarketRegime, VPINSignalStrength, VPIN_TIER1_SYMBOLS
)

# Test fixtures
from .fixtures import (
    sample_vpin_config, sample_vpin_signal, vpin_test_helper,
    level2_order_book_data, market_regime_scenarios
)
from .mock_level2_generator import MockLevel2Generator, create_mock_generator


# Create FastAPI app for testing
def create_test_app():
    """Create test FastAPI application with VPIN routes"""
    app = FastAPI(title="VPIN Test API")
    app.include_router(vpin_router, prefix="/api/v1/vpin")
    return app


@pytest.fixture
def test_client():
    """Create test client for VPIN API"""
    app = create_test_app()
    return TestClient(app)


@pytest.fixture
def mock_vpin_engine():
    """Mock VPIN engine for API testing"""
    engine = Mock(spec=VPINEngine)
    
    # Mock status response
    engine.get_status.return_value = {
        "is_running": True,
        "components_initialized": True,
        "data_feeds_active": True,
        "active_symbols": 3,
        "total_calculations": 12450,
        "last_update": 1692891200_000_000_000,
        "performance_score": 0.94,
        "error_count": 0
    }
    
    # Mock performance metrics
    engine.get_performance_metrics.return_value = {
        "vpin_calculation_time_ms": 1.8,
        "neural_analysis_time_ms": 4.2,
        "gpu_utilization": 0.85,
        "neural_engine_utilization": 0.72,
        "memory_usage_mb": 256,
        "calculations_per_second": 1250
    }
    
    return engine


class TestVPINHealthEndpoints:
    """Test VPIN health and status endpoints"""
    
    def test_health_endpoint(self, test_client, mock_vpin_engine):
        """Test VPIN engine health endpoint"""
        with patch('backend.engines.vpin.routes.vpin_engine', mock_vpin_engine):
            response = test_client.get("/api/v1/vpin/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "status" in data
            assert "timestamp" in data
            assert "components" in data
            assert "performance" in data
            
            # Verify component status
            components = data["components"]
            assert "level2_collector" in components
            assert "volume_synchronizer" in components
            assert "gpu_calculator" in components
            assert "neural_analyzer" in components
            
    def test_metrics_endpoint(self, test_client, mock_vpin_engine):
        """Test performance metrics endpoint"""
        with patch('backend.engines.vpin.routes.vpin_engine', mock_vpin_engine):
            response = test_client.get("/api/v1/vpin/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify metrics structure
            assert "vpin_calculation_time_ms" in data
            assert "gpu_utilization" in data
            assert "neural_engine_utilization" in data
            assert "calculations_per_second" in data
            
            # Verify performance values
            assert data["vpin_calculation_time_ms"] < 5.0  # Should be under 5ms
            assert data["gpu_utilization"] > 0.0
            

class TestVPINRealtimeEndpoints:
    """Test real-time VPIN calculation endpoints"""
    
    def test_realtime_vpin_single_symbol(self, test_client, mock_vpin_engine):
        """Test real-time VPIN for single symbol"""
        symbol = "AAPL"
        
        # Mock real-time VPIN response
        mock_response = {
            "symbol": symbol,
            "timestamp": 1692891200_000_000_000,
            "vpin_value": 0.42,
            "toxicity_level": "MODERATE",
            "confidence": 0.87,
            "volume_bucket": {
                "bucket_id": 1523,
                "target_volume": 100000.0,
                "buy_volume": 58400.0,
                "sell_volume": 41600.0,
                "order_imbalance": 0.168
            },
            "market_regime": "NORMAL_VOLATILITY",
            "neural_patterns": {
                "informed_trading_probability": 0.42,
                "adverse_selection_risk": "MODERATE",
                "predicted_direction": "BULLISH"
            }
        }
        
        with patch('backend.engines.vpin.routes.get_realtime_vpin_signal') as mock_realtime:
            mock_realtime.return_value = mock_response
            
            response = test_client.get(f"/api/v1/vpin/realtime/{symbol}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == symbol
            assert "vpin_value" in data
            assert "toxicity_level" in data
            assert "market_regime" in data
            assert "volume_bucket" in data
            assert "neural_patterns" in data
            
            # Validate VPIN value range
            assert 0.0 <= data["vpin_value"] <= 1.0
            
    def test_realtime_vpin_invalid_symbol(self, test_client):
        """Test real-time VPIN with invalid symbol"""
        response = test_client.get("/api/v1/vpin/realtime/INVALID")
        
        # Should return 404 or 400 for invalid symbol
        assert response.status_code in [400, 404]
        
    def test_realtime_vpin_batch(self, test_client, mock_vpin_engine):
        """Test batch real-time VPIN for multiple symbols"""
        symbols = "AAPL,TSLA,MSFT"
        
        mock_responses = [
            {
                "symbol": "AAPL",
                "vpin_value": 0.42,
                "toxicity_level": "MODERATE",
                "market_regime": "NORMAL"
            },
            {
                "symbol": "TSLA", 
                "vpin_value": 0.68,
                "toxicity_level": "HIGH",
                "market_regime": "STRESSED"
            },
            {
                "symbol": "MSFT",
                "vpin_value": 0.31,
                "toxicity_level": "LOW", 
                "market_regime": "NORMAL"
            }
        ]
        
        with patch('backend.engines.vpin.routes.get_batch_realtime_vpin') as mock_batch:
            mock_batch.return_value = mock_responses
            
            response = test_client.get(f"/api/v1/vpin/realtime/batch?symbols={symbols}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 3
            
            # Verify each symbol response
            symbols_returned = [item["symbol"] for item in data]
            assert "AAPL" in symbols_returned
            assert "TSLA" in symbols_returned
            assert "MSFT" in symbols_returned
            
    def test_realtime_vpin_batch_limit(self, test_client):
        """Test batch VPIN with too many symbols (should limit to 8)"""
        # Request 10 symbols (over limit of 8)
        symbols = ",".join([f"SYM{i}" for i in range(10)])
        
        response = test_client.get(f"/api/v1/vpin/realtime/batch?symbols={symbols}")
        
        # Should return error or limit to 8 symbols
        if response.status_code == 200:
            data = response.json()
            assert len(data) <= 8
        else:
            assert response.status_code == 400


class TestVPINHistoricalEndpoints:
    """Test historical VPIN data endpoints"""
    
    def test_historical_vpin_data(self, test_client, mock_vpin_engine):
        """Test historical VPIN data retrieval"""
        symbol = "AAPL"
        start_date = "2025-08-20T09:30:00Z"
        end_date = "2025-08-24T16:00:00Z"
        
        mock_response = {
            "symbol": symbol,
            "period": {
                "start": start_date,
                "end": end_date
            },
            "bucket_size": 100000,
            "data_points": 1247,
            "vpin_history": [
                {
                    "timestamp": "2025-08-20T09:30:15Z",
                    "vpin_value": 0.38,
                    "toxicity_level": "LOW",
                    "volume_bucket_id": 1450
                },
                {
                    "timestamp": "2025-08-20T09:31:45Z", 
                    "vpin_value": 0.45,
                    "toxicity_level": "MODERATE",
                    "volume_bucket_id": 1451
                }
            ],
            "statistics": {
                "mean_vpin": 0.41,
                "volatility": 0.15,
                "high_toxicity_periods": 23,
                "informed_trading_events": 7
            }
        }
        
        with patch('backend.engines.vpin.routes.get_historical_vpin_data') as mock_history:
            mock_history.return_value = mock_response
            
            response = test_client.get(
                f"/api/v1/vpin/history/{symbol}?"
                f"start_date={start_date}&end_date={end_date}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == symbol
            assert "vpin_history" in data
            assert "statistics" in data
            assert data["data_points"] > 0
            
            # Verify statistics
            stats = data["statistics"]
            assert "mean_vpin" in stats
            assert "volatility" in stats
            assert "high_toxicity_periods" in stats
            
    def test_historical_vpin_invalid_date_range(self, test_client):
        """Test historical VPIN with invalid date range"""
        symbol = "AAPL"
        start_date = "2025-08-25"  # Future date
        end_date = "2025-08-24"   # Before start date
        
        response = test_client.get(
            f"/api/v1/vpin/history/{symbol}?"
            f"start_date={start_date}&end_date={end_date}"
        )
        
        assert response.status_code == 400


class TestVPINAlertEndpoints:
    """Test toxicity alert endpoints"""
    
    def test_toxicity_alerts_active(self, test_client, mock_vpin_engine):
        """Test active toxicity alerts retrieval"""
        symbol = "AAPL"
        
        mock_response = {
            "symbol": symbol,
            "active_alerts": [
                {
                    "alert_id": "alert_1692891234",
                    "type": "HIGH_TOXICITY",
                    "severity": "WARNING",
                    "vpin_threshold": 0.7,
                    "current_vpin": 0.73,
                    "triggered_at": "2025-08-24T11:45:23Z",
                    "duration_ms": 15000,
                    "description": "VPIN above 0.7 indicating high informed trading probability"
                }
            ],
            "alert_history_24h": 3
        }
        
        with patch('backend.engines.vpin.routes.get_toxicity_alerts') as mock_alerts:
            mock_alerts.return_value = mock_response
            
            response = test_client.get(f"/api/v1/vpin/alerts/toxicity?symbol={symbol}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list) or "active_alerts" in data
            if "active_alerts" in data:
                assert data["symbol"] == symbol
                assert isinstance(data["active_alerts"], list)
                
    def test_alert_subscription(self, test_client, mock_vpin_engine):
        """Test toxicity alert subscription"""
        symbol = "AAPL"
        subscription_data = {
            "thresholds": {
                "high_toxicity": 0.65,
                "extreme_toxicity": 0.8
            },
            "alert_types": ["HIGH_TOXICITY", "REGIME_CHANGE", "INFORMED_TRADING"],
            "notification_method": "WEBSOCKET"
        }
        
        with patch('backend.engines.vpin.routes.subscribe_to_alerts') as mock_subscribe:
            mock_subscribe.return_value = {"status": "subscribed", "subscription_id": "sub_123"}
            
            response = test_client.post(
                f"/api/v1/vpin/alerts/{symbol}/subscribe",
                json=subscription_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert data["status"] == "subscribed"


class TestVPINPatternAnalysisEndpoints:
    """Test neural pattern analysis endpoints"""
    
    def test_pattern_regime_analysis(self, test_client, mock_vpin_engine):
        """Test market regime pattern analysis"""
        symbol = "AAPL"
        
        mock_response = {
            "symbol": symbol,
            "timestamp": "2025-08-24T12:00:00Z",
            "pattern_analysis": {
                "market_regime": "NORMAL_VOLATILITY",
                "regime_confidence": 0.89,
                "regime_duration_ms": 45000,
                "pattern_features": {
                    "volatility_cluster": False,
                    "informed_trading_cluster": True,
                    "order_flow_imbalance": "MODERATE_SELL",
                    "liquidity_provision_pattern": "NORMAL"
                },
                "predictions": {
                    "next_regime_probability": {
                        "HIGH_VOLATILITY": 0.23,
                        "LOW_VOLATILITY": 0.12,
                        "NORMAL_VOLATILITY": 0.65
                    },
                    "price_direction": {
                        "bullish_probability": 0.31,
                        "bearish_probability": 0.69
                    }
                }
            },
            "neural_engine_metrics": {
                "inference_time_ms": 4.1,
                "model_confidence": 0.87,
                "hardware_acceleration": "NEURAL_ENGINE"
            }
        }
        
        with patch('backend.engines.vpin.routes.get_pattern_analysis') as mock_patterns:
            mock_patterns.return_value = mock_response
            
            response = test_client.get(f"/api/v1/vpin/patterns/regime/{symbol}")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == symbol
            assert "pattern_analysis" in data
            assert "neural_engine_metrics" in data
            
            # Verify neural engine metrics
            metrics = data["neural_engine_metrics"]
            assert "inference_time_ms" in metrics
            assert metrics["inference_time_ms"] < 10.0  # Should be fast
            assert "hardware_acceleration" in metrics


class TestVPINOrderBookEndpoints:
    """Test Level 2 order book endpoints"""
    
    def test_order_book_depth(self, test_client, level2_order_book_data):
        """Test Level 2 order book depth retrieval"""
        symbol = "AAPL"
        
        with patch('backend.engines.vpin.routes.get_order_book_depth') as mock_depth:
            mock_depth.return_value = level2_order_book_data
            
            response = test_client.get(f"/api/v1/vpin/orderbook/{symbol}/depth")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["symbol"] == symbol
            assert "bids" in data
            assert "asks" in data
            assert "timestamp" in data
            
            # Verify bid/ask structure
            assert len(data["bids"]) > 0
            assert len(data["asks"]) > 0
            
            # Verify level structure
            for bid in data["bids"]:
                assert "price" in bid
                assert "size" in bid
                assert "exchange" in bid
                assert "mm" in bid  # Market maker
                
    def test_symbol_subscription(self, test_client, mock_vpin_engine):
        """Test Level 2 data subscription for symbol"""
        symbol = "AAPL"
        
        with patch('backend.engines.vpin.routes.subscribe_to_level2_data') as mock_sub:
            mock_sub.return_value = {"status": "subscribed", "symbol": symbol}
            
            response = test_client.post(f"/api/v1/vpin/symbols/{symbol}/subscribe")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "subscribed"
            assert data["symbol"] == symbol
            
    def test_symbol_unsubscription(self, test_client, mock_vpin_engine):
        """Test Level 2 data unsubscription"""
        symbol = "AAPL"
        
        with patch('backend.engines.vpin.routes.unsubscribe_from_level2_data') as mock_unsub:
            mock_unsub.return_value = {"status": "unsubscribed", "symbol": symbol}
            
            response = test_client.delete(f"/api/v1/vpin/symbols/{symbol}/unsubscribe")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unsubscribed"
            assert data["symbol"] == symbol


class TestVPINConfigurationEndpoints:
    """Test VPIN configuration endpoints"""
    
    def test_get_config(self, test_client, mock_vpin_engine):
        """Test VPIN configuration retrieval"""
        mock_config = {
            "default_bucket_size": 100000,
            "enable_neural_analysis": True,
            "gpu_acceleration": True,
            "alert_thresholds": {
                "moderate_toxicity": 0.5,
                "high_toxicity": 0.7,
                "extreme_toxicity": 0.85
            },
            "supported_symbols": VPIN_TIER1_SYMBOLS
        }
        
        with patch('backend.engines.vpin.routes.get_vpin_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            
            response = test_client.get("/api/v1/vpin/config")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "default_bucket_size" in data
            assert "enable_neural_analysis" in data
            assert "supported_symbols" in data
            assert len(data["supported_symbols"]) == 8
            
    def test_update_config(self, test_client, mock_vpin_engine):
        """Test VPIN configuration update"""
        new_config = {
            "default_bucket_size": 50000,
            "alert_thresholds": {
                "high_toxicity": 0.65,
                "extreme_toxicity": 0.8
            }
        }
        
        with patch('backend.engines.vpin.routes.update_vpin_config') as mock_update:
            mock_update.return_value = {"status": "updated", "config": new_config}
            
            response = test_client.post("/api/v1/vpin/config/update", json=new_config)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "updated"
            assert "config" in data


class TestVPINWebSocketEndpoints:
    """Test VPIN WebSocket endpoints"""
    
    def test_vpin_websocket_connection(self, test_client):
        """Test VPIN WebSocket connection"""
        symbol = "AAPL"
        
        with test_client.websocket_connect(f"/api/v1/vpin/ws/vpin/{symbol}") as websocket:
            # Send subscription message
            websocket.send_json({"action": "subscribe", "symbol": symbol})
            
            # Should receive confirmation
            data = websocket.receive_json()
            assert "status" in data
            assert data["status"] in ["connected", "subscribed"]
            
    def test_order_book_websocket_connection(self, test_client):
        """Test Level 2 order book WebSocket connection"""
        symbol = "AAPL"
        
        with test_client.websocket_connect(f"/api/v1/vpin/ws/orderbook/{symbol}") as websocket:
            # Send subscription message
            websocket.send_json({"action": "subscribe", "symbol": symbol})
            
            # Should receive confirmation or initial data
            data = websocket.receive_json()
            assert "status" in data or "symbol" in data
            
    @pytest.mark.asyncio
    async def test_websocket_vpin_streaming(self, test_client):
        """Test real-time VPIN streaming via WebSocket"""
        symbol = "AAPL"
        
        # Mock WebSocket streaming
        async def mock_stream():
            return {
                "symbol": symbol,
                "vpin_value": 0.42,
                "timestamp": 1692891200_000_000_000,
                "toxicity_level": "MODERATE"
            }
        
        # This would require more complex WebSocket mocking for full implementation
        # For now, we test the connection establishment above


class TestVPINDataQualityEndpoints:
    """Test VPIN data quality monitoring endpoints"""
    
    def test_data_quality_metrics(self, test_client, mock_vpin_engine):
        """Test data quality metrics retrieval"""
        mock_response = {
            "AAPL": {
                "symbol": "AAPL",
                "completeness_score": 0.95,
                "latency_score": 0.88,
                "accuracy_score": 0.92,
                "last_update": "2025-08-24T12:00:00Z",
                "issues": []
            },
            "TSLA": {
                "symbol": "TSLA", 
                "completeness_score": 0.87,
                "latency_score": 0.91,
                "accuracy_score": 0.89,
                "last_update": "2025-08-24T12:00:00Z",
                "issues": ["intermittent_feed_delay"]
            }
        }
        
        with patch('backend.engines.vpin.routes.get_data_quality_metrics') as mock_quality:
            mock_quality.return_value = mock_response
            
            response = test_client.get("/api/v1/vpin/data-quality")
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            for symbol, quality in data.items():
                assert "completeness_score" in quality
                assert "latency_score" in quality
                assert "accuracy_score" in quality
                assert 0.0 <= quality["completeness_score"] <= 1.0


class TestVPINPerformanceEndpoints:
    """Test VPIN performance monitoring endpoints"""
    
    def test_performance_metrics_detailed(self, test_client, mock_vpin_engine):
        """Test detailed performance metrics"""
        mock_response = {
            "system_metrics": {
                "vpin_calculation_time_ms": 1.8,
                "neural_analysis_time_ms": 4.2,
                "level2_processing_time_ms": 0.8,
                "total_processing_time_ms": 6.8
            },
            "hardware_metrics": {
                "gpu_utilization": 0.85,
                "neural_engine_utilization": 0.72,
                "cpu_utilization": 0.34,
                "memory_usage_mb": 256
            },
            "throughput_metrics": {
                "calculations_per_second": 1250,
                "trades_processed_per_second": 450,
                "quotes_processed_per_second": 1800
            },
            "quality_metrics": {
                "average_data_quality": 0.91,
                "classification_accuracy": 0.95,
                "alert_precision": 0.88
            }
        }
        
        with patch('backend.engines.vpin.routes.get_performance_metrics') as mock_perf:
            mock_perf.return_value = mock_response
            
            response = test_client.get("/api/v1/vpin/performance/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "system_metrics" in data
            assert "hardware_metrics" in data
            assert "throughput_metrics" in data
            assert "quality_metrics" in data
            
            # Verify performance targets
            system = data["system_metrics"]
            assert system["vpin_calculation_time_ms"] < 5.0
            assert system["neural_analysis_time_ms"] < 10.0


class TestVPINErrorHandling:
    """Test VPIN API error handling"""
    
    def test_invalid_symbol_error(self, test_client):
        """Test handling of invalid symbols"""
        response = test_client.get("/api/v1/vpin/realtime/INVALID_SYMBOL")
        assert response.status_code in [400, 404]
        
        if response.status_code == 400:
            data = response.json()
            assert "detail" in data or "error" in data
            
    def test_missing_required_parameters(self, test_client):
        """Test handling of missing required parameters"""
        # Test historical endpoint without dates
        response = test_client.get("/api/v1/vpin/history/AAPL")
        # Should work with default dates or return 400
        assert response.status_code in [200, 400]
        
    def test_invalid_date_format(self, test_client):
        """Test handling of invalid date formats"""
        response = test_client.get(
            "/api/v1/vpin/history/AAPL?start_date=invalid_date&end_date=also_invalid"
        )
        assert response.status_code == 400
        
    def test_server_error_handling(self, test_client):
        """Test handling of server errors"""
        with patch('backend.engines.vpin.routes.get_realtime_vpin_signal') as mock_error:
            mock_error.side_effect = Exception("Simulated server error")
            
            response = test_client.get("/api/v1/vpin/realtime/AAPL")
            assert response.status_code == 500


# Integration tests combining multiple endpoints
class TestVPINAPIIntegration:
    """Test VPIN API integration scenarios"""
    
    def test_complete_vpin_workflow(self, test_client, mock_vpin_engine):
        """Test complete VPIN analysis workflow"""
        symbol = "AAPL"
        
        # 1. Subscribe to symbol
        with patch('backend.engines.vpin.routes.subscribe_to_level2_data') as mock_sub:
            mock_sub.return_value = {"status": "subscribed"}
            response = test_client.post(f"/api/v1/vpin/symbols/{symbol}/subscribe")
            assert response.status_code == 200
        
        # 2. Get real-time VPIN
        with patch('backend.engines.vpin.routes.get_realtime_vpin_signal') as mock_realtime:
            mock_realtime.return_value = {
                "symbol": symbol,
                "vpin_value": 0.42,
                "toxicity_level": "MODERATE"
            }
            response = test_client.get(f"/api/v1/vpin/realtime/{symbol}")
            assert response.status_code == 200
            
        # 3. Check for alerts
        with patch('backend.engines.vpin.routes.get_toxicity_alerts') as mock_alerts:
            mock_alerts.return_value = {"active_alerts": []}
            response = test_client.get(f"/api/v1/vpin/alerts/toxicity?symbol={symbol}")
            assert response.status_code == 200
            
        # 4. Get performance metrics
        with patch('backend.engines.vpin.routes.get_performance_metrics') as mock_perf:
            mock_perf.return_value = {"vpin_calculation_time_ms": 1.8}
            response = test_client.get("/api/v1/vpin/performance/metrics")
            assert response.status_code == 200
            
    def test_multi_symbol_management(self, test_client, mock_vpin_engine):
        """Test managing multiple symbols simultaneously"""
        symbols = ["AAPL", "TSLA", "MSFT"]
        
        # Subscribe to multiple symbols
        for symbol in symbols:
            with patch('backend.engines.vpin.routes.subscribe_to_level2_data'):
                response = test_client.post(f"/api/v1/vpin/symbols/{symbol}/subscribe")
                assert response.status_code == 200
                
        # Get batch real-time data
        with patch('backend.engines.vpin.routes.get_batch_realtime_vpin') as mock_batch:
            mock_batch.return_value = [
                {"symbol": s, "vpin_value": 0.4 + i*0.1} 
                for i, s in enumerate(symbols)
            ]
            
            symbols_param = ",".join(symbols)
            response = test_client.get(f"/api/v1/vpin/realtime/batch?symbols={symbols_param}")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) == len(symbols)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
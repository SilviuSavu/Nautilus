"""
Integration tests for Data Catalog and Pipeline functionality
Tests the complete data pipeline from API endpoints to Nautilus integration
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from data_catalog_routes import execute_nautilus_command

# Test client
client = TestClient(app)

class TestDataCatalogIntegration:
    """Test data catalog API endpoints and integration"""
    
    def test_get_data_catalog_success(self):
        """Test successful data catalog retrieval"""
        response = client.get("/api/v1/nautilus/data/catalog")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify catalog structure
        assert "instruments" in data
        assert "venues" in data
        assert "dataSources" in data
        assert "qualityMetrics" in data
        assert "lastUpdated" in data
        assert "totalInstruments" in data
        assert "totalRecords" in data
        assert "storageSize" in data
        
        # Verify instruments have required fields
        if data["instruments"]:
            instrument = data["instruments"][0]
            assert "instrumentId" in instrument
            assert "venue" in instrument
            assert "symbol" in instrument
            assert "qualityScore" in instrument
            assert "recordCount" in instrument
    
    def test_get_instrument_data_success(self):
        """Test successful instrument data retrieval"""
        instrument_id = "EURUSD.SIM"
        response = client.get(f"/api/v1/nautilus/data/catalog/instruments/{instrument_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify instrument structure
        assert data["instrumentId"] == instrument_id
        assert "venue" in data
        assert "symbol" in data
        assert "assetClass" in data
        assert "dataType" in data
        assert "qualityScore" in data
        assert "dateRange" in data
        assert "start" in data["dateRange"]
        assert "end" in data["dateRange"]
    
    def test_get_instrument_data_not_found(self):
        """Test instrument not found error"""
        response = client.get("/api/v1/nautilus/data/catalog/instruments/INVALID.SIM")
        
        # Should return 404 or handle gracefully
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_200_OK]
    
    def test_analyze_data_gaps(self):
        """Test data gap analysis"""
        instrument_id = "EURUSD.SIM"
        response = client.get(f"/api/v1/nautilus/data/gaps/{instrument_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "gaps" in data
        # Gaps should be a list
        assert isinstance(data["gaps"], list)
        
        # If gaps exist, verify structure
        for gap in data["gaps"]:
            assert "id" in gap
            assert "start" in gap
            assert "end" in gap
            assert "severity" in gap
            assert gap["severity"] in ["low", "medium", "high"]
    
    def test_get_quality_metrics(self):
        """Test quality metrics retrieval"""
        instrument_id = "EURUSD.SIM"
        response = client.get(f"/api/v1/nautilus/data/quality/{instrument_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify metrics structure
        assert "completeness" in data
        assert "accuracy" in data
        assert "timeliness" in data
        assert "consistency" in data
        assert "overall" in data
        assert "lastUpdated" in data
        
        # Verify metric values are between 0 and 1
        for metric in ["completeness", "accuracy", "timeliness", "consistency", "overall"]:
            assert 0 <= data[metric] <= 1
    
    def test_validate_data_quality(self):
        """Test data quality validation"""
        request_data = {"instrumentId": "EURUSD.SIM"}
        response = client.post("/api/v1/nautilus/data/quality/validate", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["instrumentId"] == "EURUSD.SIM"
        assert "metrics" in data
        assert "anomalies" in data
        assert "validationResults" in data
        assert "generatedAt" in data
    
    def test_validate_data_quality_missing_instrument(self):
        """Test validation with missing instrument ID"""
        response = client.post("/api/v1/nautilus/data/quality/validate", json={})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_export_data_success(self):
        """Test successful data export"""
        export_request = {
            "instrumentIds": ["EURUSD.SIM"],
            "format": "parquet",
            "dateRange": {
                "start": "2024-01-01",
                "end": "2024-01-31"
            },
            "compression": True,
            "includeMetadata": True
        }
        
        response = client.post("/api/v1/nautilus/data/export", json=export_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "exportId" in data
        assert "success" in data
        assert "format" in data
        assert data["format"] == "parquet"
        assert "createdAt" in data
    
    def test_export_data_invalid_format(self):
        """Test export with invalid format"""
        export_request = {
            "instrumentIds": ["EURUSD.SIM"],
            "format": "invalid_format",
            "dateRange": {
                "start": "2024-01-01",
                "end": "2024-01-31"
            }
        }
        
        response = client.post("/api/v1/nautilus/data/export", json=export_request)
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_import_data_success(self):
        """Test successful data import"""
        import_request = {
            "filePath": "/tmp/test_data.csv",
            "format": "csv",
            "instrumentId": "CUSTOM.SIM",
            "venue": "SIM",
            "dataType": "bar",
            "validateData": True
        }
        
        response = client.post("/api/v1/nautilus/data/import", json=import_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "importId" in data
        assert "success" in data
        assert "processedAt" in data
    
    def test_get_feed_statuses(self):
        """Test feed status retrieval"""
        response = client.get("/api/v1/nautilus/data/feeds/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "feeds" in data
        assert isinstance(data["feeds"], list)
        
        # Verify feed structure if feeds exist
        for feed in data["feeds"]:
            assert "feedId" in feed
            assert "source" in feed
            assert "status" in feed
            assert feed["status"] in ["connected", "disconnected", "degraded", "reconnecting"]
            assert "latency" in feed
            assert "throughput" in feed
            assert "qualityScore" in feed
    
    def test_subscribe_feed_success(self):
        """Test successful feed subscription"""
        request_data = {
            "feedId": "ib_market_data",
            "instrumentIds": ["EURUSD.SIM", "GBPUSD.SIM"]
        }
        
        response = client.post("/api/v1/nautilus/data/feeds/subscribe", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
    
    def test_subscribe_feed_missing_data(self):
        """Test feed subscription with missing data"""
        response = client.post("/api/v1/nautilus/data/feeds/subscribe", json={})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_unsubscribe_feed_success(self):
        """Test successful feed unsubscription"""
        request_data = {
            "feedId": "ib_market_data",
            "instrumentIds": ["EURUSD.SIM"]
        }
        
        response = client.delete("/api/v1/nautilus/data/feeds/unsubscribe", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
    
    def test_get_pipeline_health(self):
        """Test pipeline health status"""
        response = client.get("/api/v1/nautilus/data/pipeline/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "critical"]
        assert "details" in data
    
    def test_search_instruments(self):
        """Test instrument search functionality"""
        params = {
            "venue": "SIM",
            "assetClass": "Currency",
            "dataType": "tick"
        }
        
        response = client.get("/api/v1/nautilus/data/catalog/search", params=params)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "instruments" in data
        assert "totalCount" in data
        assert "filters" in data
        
        # Verify filters were applied
        assert data["filters"]["venue"] == "SIM"
        assert data["filters"]["assetClass"] == "Currency"
        assert data["filters"]["dataType"] == "tick"


class TestNautilusDockerIntegration:
    """Test Docker integration with Nautilus"""
    
    @patch('data_catalog_routes.asyncio.create_subprocess_exec')
    async def test_execute_nautilus_command_success(self, mock_subprocess):
        """Test successful Nautilus command execution"""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b'{"instruments": ["EURUSD.SIM"], "total": 1}',
            b''
        )
        mock_subprocess.return_value = mock_process
        
        result = await execute_nautilus_command("test_command")
        
        assert "instruments" in result
        assert result["total"] == 1
    
    @patch('data_catalog_routes.asyncio.create_subprocess_exec')
    async def test_execute_nautilus_command_error(self, mock_subprocess):
        """Test Nautilus command execution error"""
        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (
            b'',
            b'Command failed'
        )
        mock_subprocess.return_value = mock_process
        
        with pytest.raises(Exception):
            await execute_nautilus_command("invalid_command")
    
    def test_docker_execute_endpoint(self):
        """Test Docker command execution endpoint"""
        request_data = {
            "command": "catalog",
            "args": "list_instruments"
        }
        
        response = client.post("/api/v1/nautilus/docker/execute", json=request_data)
        
        # Should return 200 or appropriate error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    def test_docker_catalog_summary(self):
        """Test Docker catalog summary endpoint"""
        response = client.get("/api/v1/nautilus/docker/catalog/summary")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "instruments" in data
        assert "totalRecords" in data
        assert "storageSize" in data


class TestDataPipelinePerformance:
    """Test data pipeline performance and scalability"""
    
    def test_large_catalog_response_time(self):
        """Test catalog response time with large datasets"""
        start_time = datetime.now()
        
        response = client.get("/api/v1/nautilus/data/catalog")
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        # Should respond within 3 seconds per story requirement
        assert response_time < 3.0
        assert response.status_code == status.HTTP_200_OK
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/nautilus/data/feeds/status")
            results.append(response.status_code)
        
        # Create multiple concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status_code == 200 for status_code in results)
    
    def test_export_large_dataset_simulation(self):
        """Test export of large dataset (simulated)"""
        export_request = {
            "instrumentIds": ["EURUSD.SIM"],
            "format": "parquet",
            "dateRange": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            },
            "maxRecords": 1000000  # 1M records
        }
        
        start_time = datetime.now()
        response = client.post("/api/v1/nautilus/data/export", json=export_request)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert response.status_code == status.HTTP_200_OK
        # Export should complete within reasonable time
        assert processing_time < 10.0


class TestDataQualityValidation:
    """Test data quality validation and monitoring"""
    
    def test_quality_metrics_validation(self):
        """Test quality metrics return valid values"""
        response = client.get("/api/v1/nautilus/data/quality/EURUSD.SIM")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # All metrics should be between 0 and 1
        metrics = ["completeness", "accuracy", "timeliness", "consistency", "overall"]
        for metric in metrics:
            assert metric in data
            assert 0 <= data[metric] <= 1
    
    def test_gap_analysis_comprehensive(self):
        """Test comprehensive gap analysis"""
        response = client.get("/api/v1/nautilus/data/gaps/EURUSD.SIM")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify gap structure
        for gap in data.get("gaps", []):
            # Gap should have valid datetime format
            start_time = datetime.fromisoformat(gap["start"].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(gap["end"].replace('Z', '+00:00'))
            
            # End time should be after start time
            assert end_time > start_time
            
            # Severity should be valid
            assert gap["severity"] in ["low", "medium", "high"]


class TestErrorHandling:
    """Test error handling and resilience"""
    
    def test_invalid_instrument_id_format(self):
        """Test handling of invalid instrument ID format"""
        response = client.get("/api/v1/nautilus/data/catalog/instruments/INVALID_FORMAT")
        
        # Should handle gracefully without crashing
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_200_OK
        ]
    
    def test_malformed_export_request(self):
        """Test handling of malformed export request"""
        malformed_request = {
            "instrumentIds": [],  # Empty list
            "format": "invalid",
            "dateRange": {
                "start": "invalid-date",
                "end": "2024-01-31"
            }
        }
        
        response = client.post("/api/v1/nautilus/data/export", json=malformed_request)
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_api_resilience_under_load(self):
        """Test API resilience under high load"""
        import time
        
        # Make rapid successive requests
        responses = []
        for _ in range(50):
            response = client.get("/api/v1/nautilus/data/pipeline/health")
            responses.append(response.status_code)
            time.sleep(0.01)  # Small delay
        
        # Most requests should succeed
        success_rate = sum(1 for status_code in responses if status_code == 200) / len(responses)
        assert success_rate >= 0.9  # At least 90% success rate


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations and background tasks"""
    
    async def test_async_export_processing(self):
        """Test asynchronous export processing"""
        # This would test background task processing
        # For now, we'll test the endpoint response
        export_request = {
            "instrumentIds": ["EURUSD.SIM"],
            "format": "parquet",
            "dateRange": {
                "start": "2024-01-01",
                "end": "2024-01-31"
            }
        }
        
        response = client.post("/api/v1/nautilus/data/export", json=export_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should return export ID for tracking
        assert "exportId" in data
    
    async def test_real_time_feed_monitoring(self):
        """Test real-time feed monitoring capabilities"""
        # Test that feed status endpoint returns current data
        response = client.get("/api/v1/nautilus/data/feeds/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Feed data should include timestamp information
        for feed in data.get("feeds", []):
            assert "lastUpdate" in feed
            # Timestamp should be parseable
            last_update = datetime.fromisoformat(feed["lastUpdate"].replace('Z', '+00:00'))
            assert isinstance(last_update, datetime)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
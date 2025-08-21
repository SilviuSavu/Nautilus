"""
Performance tests for Data Catalog and Pipeline functionality
Tests performance requirements and scalability under load
"""

import pytest
import asyncio
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import status
import psutil
import gc

from main import app

# Test client
client = TestClient(app)

class TestDataCatalogPerformance:
    """Performance tests for data catalog endpoints"""
    
    def test_catalog_response_time_requirement(self):
        """Test catalog loading time meets <3 seconds requirement"""
        start_time = time.time()
        
        response = client.get("/api/v1/nautilus/data/catalog")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Story requirement: <3 seconds for 1000+ instruments
        assert response_time < 3.0
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # Should handle reasonable dataset size
        assert data["totalInstruments"] >= 0
    
    def test_large_instrument_search_performance(self):
        """Test search performance with various filter combinations"""
        search_params = [
            {"venue": "SIM"},
            {"assetClass": "Currency"},
            {"dataType": "tick"},
            {"venue": "SIM", "assetClass": "Currency"},
            {"qualityThreshold": "0.9"},
            {"venue": "SIM", "assetClass": "Currency", "dataType": "tick", "qualityThreshold": "0.8"}
        ]
        
        response_times = []
        
        for params in search_params:
            start_time = time.time()
            
            response = client.get("/api/v1/nautilus/data/catalog/search", params=params)
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response.status_code == status.HTTP_200_OK
            assert response_time < 2.0  # Search should be fast
        
        # Average search time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0
    
    def test_concurrent_catalog_requests(self):
        """Test handling of concurrent catalog requests"""
        num_threads = 20
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/api/v1/nautilus/data/catalog")
            end_time = time.time()
            
            results.append({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        # Create and start threads
        threads = []
        start_time = time.time()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        total_time = time.time() - start_time
        
        # All requests should succeed
        assert len(results) == num_threads
        assert all(r['status_code'] == 200 for r in results)
        
        # Concurrent processing should be efficient
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        assert avg_response_time < 5.0
        assert total_time < 15.0  # Total time for all concurrent requests
    
    def test_memory_usage_under_load(self):
        """Test memory usage remains stable under load"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests to stress test memory usage
        for _ in range(100):
            response = client.get("/api/v1/nautilus/data/catalog")
            assert response.status_code == status.HTTP_200_OK
            
            # Occasionally force garbage collection
            if _ % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100
    
    def test_export_large_dataset_simulation(self):
        """Test export performance with large dataset simulation"""
        export_requests = [
            {
                "instrumentIds": [f"INST{i}.SIM" for i in range(10)],
                "format": "parquet",
                "dateRange": {"start": "2024-01-01", "end": "2024-01-31"},
                "maxRecords": 100000
            },
            {
                "instrumentIds": [f"INST{i}.SIM" for i in range(50)],
                "format": "csv",
                "dateRange": {"start": "2024-01-01", "end": "2024-03-31"},
                "maxRecords": 500000
            },
            {
                "instrumentIds": [f"INST{i}.SIM" for i in range(100)],
                "format": "parquet",
                "dateRange": {"start": "2024-01-01", "end": "2024-12-31"},
                "maxRecords": 1000000
            }
        ]
        
        for i, request_data in enumerate(export_requests):
            start_time = time.time()
            
            response = client.post("/api/v1/nautilus/data/export", json=request_data)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            
            # Export initiation should be fast regardless of dataset size
            assert response_time < 2.0
            
            # Verify export metadata
            assert "exportId" in data
            assert data["format"] == request_data["format"]


class TestQualityAnalysisPerformance:
    """Performance tests for data quality analysis"""
    
    def test_quality_validation_speed(self):
        """Test quality validation performance"""
        instruments = [f"INST{i}.SIM" for i in range(50)]
        
        validation_times = []
        
        for instrument in instruments:
            start_time = time.time()
            
            response = client.post("/api/v1/nautilus/data/quality/validate", 
                                 json={"instrumentId": instrument})
            
            end_time = time.time()
            validation_time = end_time - start_time
            validation_times.append(validation_time)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Individual validation should be fast
            assert validation_time < 1.0
        
        # Average validation time should be very fast
        avg_validation_time = sum(validation_times) / len(validation_times)
        assert avg_validation_time < 0.5
    
    def test_gap_analysis_performance(self):
        """Test gap analysis performance across multiple instruments"""
        instruments = [f"INST{i}.SIM" for i in range(30)]
        
        gap_analysis_times = []
        
        for instrument in instruments:
            start_time = time.time()
            
            response = client.get(f"/api/v1/nautilus/data/gaps/{instrument}")
            
            end_time = time.time()
            analysis_time = end_time - start_time
            gap_analysis_times.append(analysis_time)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Gap analysis should complete within 30 seconds (story requirement)
            assert analysis_time < 30.0
        
        # Most gap analyses should be much faster
        fast_analyses = [t for t in gap_analysis_times if t < 5.0]
        assert len(fast_analyses) >= len(gap_analysis_times) * 0.8  # 80% should be fast
    
    def test_concurrent_quality_operations(self):
        """Test concurrent quality validation and gap analysis"""
        instruments = [f"INST{i}.SIM" for i in range(20)]
        
        def quality_validation_task(instrument):
            response = client.post("/api/v1/nautilus/data/quality/validate", 
                                 json={"instrumentId": instrument})
            return response.status_code == 200
        
        def gap_analysis_task(instrument):
            response = client.get(f"/api/v1/nautilus/data/gaps/{instrument}")
            return response.status_code == 200
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit quality validation tasks
            quality_futures = [
                executor.submit(quality_validation_task, inst) 
                for inst in instruments[:10]
            ]
            
            # Submit gap analysis tasks
            gap_futures = [
                executor.submit(gap_analysis_task, inst) 
                for inst in instruments[10:]
            ]
            
            # Wait for all tasks to complete
            quality_results = [f.result() for f in quality_futures]
            gap_results = [f.result() for f in gap_futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All operations should succeed
        assert all(quality_results)
        assert all(gap_results)
        
        # Concurrent operations should complete efficiently
        assert total_time < 20.0


class TestFeedMonitoringPerformance:
    """Performance tests for real-time feed monitoring"""
    
    def test_feed_status_response_time(self):
        """Test feed status retrieval performance"""
        response_times = []
        
        # Test multiple requests to measure consistency
        for _ in range(50):
            start_time = time.time()
            
            response = client.get("/api/v1/nautilus/data/feeds/status")
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Each request should be fast (real-time requirement)
            assert response_time < 1.0
        
        # Consistent performance
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.5
        
        # No outliers (95th percentile)
        sorted_times = sorted(response_times)
        percentile_95 = sorted_times[int(len(sorted_times) * 0.95)]
        assert percentile_95 < 1.0
    
    def test_pipeline_health_monitoring_overhead(self):
        """Test pipeline health monitoring performance overhead"""
        # Measure baseline performance
        baseline_times = []
        for _ in range(10):
            start_time = time.time()
            response = client.get("/api/v1/nautilus/data/catalog")
            end_time = time.time()
            baseline_times.append(end_time - start_time)
            assert response.status_code == status.HTTP_200_OK
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Measure performance with concurrent health monitoring
        health_times = []
        catalog_times = []
        
        def health_monitoring():
            for _ in range(20):
                start_time = time.time()
                response = client.get("/api/v1/nautilus/data/pipeline/health")
                end_time = time.time()
                health_times.append(end_time - start_time)
                assert response.status_code == status.HTTP_200_OK
                time.sleep(0.1)
        
        # Start health monitoring in background
        health_thread = threading.Thread(target=health_monitoring)
        health_thread.start()
        
        # Test catalog performance during monitoring
        for _ in range(10):
            start_time = time.time()
            response = client.get("/api/v1/nautilus/data/catalog")
            end_time = time.time()
            catalog_times.append(end_time - start_time)
            assert response.status_code == status.HTTP_200_OK
            time.sleep(0.2)
        
        health_thread.join()
        
        # Monitoring should not significantly impact other operations
        monitoring_avg = sum(catalog_times) / len(catalog_times)
        performance_impact = (monitoring_avg - baseline_avg) / baseline_avg
        
        # Less than 20% performance impact
        assert performance_impact < 0.2
        
        # Health monitoring itself should be fast
        health_avg = sum(health_times) / len(health_times)
        assert health_avg < 0.5
    
    def test_subscription_management_performance(self):
        """Test feed subscription/unsubscription performance"""
        feed_ids = ["ib_data", "yahoo_finance", "alpha_vantage", "quandl"]
        instruments = [f"INST{i}.SIM" for i in range(20)]
        
        subscription_times = []
        unsubscription_times = []
        
        for feed_id in feed_ids:
            # Test subscription
            start_time = time.time()
            
            response = client.post("/api/v1/nautilus/data/feeds/subscribe", json={
                "feedId": feed_id,
                "instrumentIds": instruments
            })
            
            end_time = time.time()
            subscription_time = end_time - start_time
            subscription_times.append(subscription_time)
            
            assert response.status_code == status.HTTP_200_OK
            assert subscription_time < 2.0  # Should be fast
            
            # Test unsubscription
            start_time = time.time()
            
            response = client.delete("/api/v1/nautilus/data/feeds/unsubscribe", json={
                "feedId": feed_id,
                "instrumentIds": instruments
            })
            
            end_time = time.time()
            unsubscription_time = end_time - start_time
            unsubscription_times.append(unsubscription_time)
            
            assert response.status_code == status.HTTP_200_OK
            assert unsubscription_time < 2.0  # Should be fast
        
        # Average times should be very good
        avg_sub_time = sum(subscription_times) / len(subscription_times)
        avg_unsub_time = sum(unsubscription_times) / len(unsubscription_times)
        
        assert avg_sub_time < 1.0
        assert avg_unsub_time < 1.0


class TestScalabilityLimits:
    """Test system behavior at scale limits"""
    
    def test_maximum_concurrent_connections(self):
        """Test behavior with many concurrent connections"""
        max_connections = 100
        results = []
        
        def make_concurrent_request():
            try:
                response = client.get("/api/v1/nautilus/data/pipeline/health")
                results.append({
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                })
            except Exception as e:
                results.append({
                    'status_code': 500,
                    'success': False,
                    'error': str(e)
                })
        
        # Create many concurrent connections
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_connections) as executor:
            start_time = time.time()
            
            futures = [
                executor.submit(make_concurrent_request) 
                for _ in range(max_connections)
            ]
            
            # Wait for all to complete
            concurrent.futures.wait(futures, timeout=30.0)
            
            end_time = time.time()
        
        total_time = end_time - start_time
        
        # Most connections should succeed
        success_rate = sum(1 for r in results if r['success']) / len(results)
        assert success_rate >= 0.95  # 95% success rate minimum
        
        # Should handle load reasonably
        assert total_time < 30.0
    
    def test_large_export_request_handling(self):
        """Test handling of very large export requests"""
        large_export_request = {
            "instrumentIds": [f"INST{i}.SIM" for i in range(1000)],  # 1000 instruments
            "format": "parquet",
            "dateRange": {
                "start": "2020-01-01",
                "end": "2024-12-31"
            },
            "maxRecords": 100000000,  # 100M records
            "compression": True,
            "includeMetadata": True
        }
        
        start_time = time.time()
        
        response = client.post("/api/v1/nautilus/data/export", json=large_export_request)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should accept large requests without timeout
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 5.0  # Request processing should be fast
        
        data = response.json()
        assert data["success"] is True
        assert "exportId" in data
    
    def test_system_resource_usage_monitoring(self):
        """Monitor system resources during intensive operations"""
        process = psutil.Process()
        
        # Baseline measurements
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform intensive operations
        operations = [
            lambda: client.get("/api/v1/nautilus/data/catalog"),
            lambda: client.get("/api/v1/nautilus/data/feeds/status"),
            lambda: client.post("/api/v1/nautilus/data/quality/validate", 
                              json={"instrumentId": "EURUSD.SIM"}),
            lambda: client.get("/api/v1/nautilus/data/gaps/EURUSD.SIM"),
            lambda: client.post("/api/v1/nautilus/data/export", json={
                "instrumentIds": ["EURUSD.SIM"],
                "format": "parquet",
                "dateRange": {"start": "2024-01-01", "end": "2024-01-31"}
            })
        ]
        
        # Run operations multiple times
        for _ in range(20):
            for operation in operations:
                response = operation()
                assert response.status_code == status.HTTP_200_OK
        
        # Measure final resource usage
        time.sleep(1)  # Allow CPU measurement to stabilize
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Resource usage should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200  # Less than 200MB increase
        
        # CPU usage should return to reasonable levels
        assert final_cpu < 80  # Less than 80% CPU


class TestPerformanceRegression:
    """Test for performance regressions"""
    
    def test_api_response_time_consistency(self):
        """Test that API response times are consistent over multiple runs"""
        endpoints = [
            "/api/v1/nautilus/data/catalog",
            "/api/v1/nautilus/data/feeds/status",
            "/api/v1/nautilus/data/pipeline/health"
        ]
        
        for endpoint in endpoints:
            response_times = []
            
            # Measure response times over multiple calls
            for _ in range(30):
                start_time = time.time()
                response = client.get(endpoint)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert response.status_code == status.HTTP_200_OK
                time.sleep(0.1)  # Small delay between requests
            
            # Calculate statistics
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            # Response times should be consistent
            assert max_time < avg_time * 3  # No response should be 3x average
            assert max_time - min_time < 2.0  # Range should be reasonable
            assert avg_time < 1.0  # Average should be good
    
    def test_throughput_requirements(self):
        """Test API throughput requirements"""
        # Test sustained throughput
        num_requests = 100
        start_time = time.time()
        
        successful_requests = 0
        
        for _ in range(num_requests):
            response = client.get("/api/v1/nautilus/data/pipeline/health")
            if response.status_code == 200:
                successful_requests += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        requests_per_second = successful_requests / total_time
        
        # Should handle at least 50 requests per second
        assert requests_per_second >= 50
        assert successful_requests >= num_requests * 0.98  # 98% success rate


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])
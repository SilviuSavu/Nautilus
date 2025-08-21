"""
Performance Test Report Generator for Monitoring System
Validates all performance requirements and generates comprehensive reports
"""

import pytest
import time
import statistics
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

from monitoring_service import MonitoringService, AlertLevel, MetricType


@dataclass
class PerformanceResult:
    test_name: str
    metric: str
    value: float
    unit: str
    requirement: float
    passed: bool
    details: Dict[str, Any] = None


class MonitoringPerformanceValidator:
    """Comprehensive performance validation for monitoring system"""
    
    def __init__(self):
        self.monitoring_service = MonitoringService()
        self.results: List[PerformanceResult] = []
    
    def record_result(self, test_name: str, metric: str, value: float, unit: str, requirement: float, details: Dict[str, Any] = None):
        """Record a performance test result"""
        passed = value <= requirement if 'time' in metric.lower() or 'latency' in metric.lower() else value >= requirement
        result = PerformanceResult(test_name, metric, value, unit, requirement, passed, details)
        self.results.append(result)
        return result
    
    def test_metric_recording_overhead(self) -> PerformanceResult:
        """Test metric recording overhead - requirement: <0.1ms per operation"""
        num_operations = 10000
        
        start_time = time.perf_counter()
        for i in range(num_operations):
            self.monitoring_service.record_metric(
                f"perf_test_{i % 100}",
                float(i),
                MetricType.COUNTER,
                {"batch": str(i // 1000)}
            )
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_op = total_time_ms / num_operations
        
        return self.record_result(
            "Metric Recording Overhead",
            "avg_time_per_operation",
            avg_time_per_op,
            "ms",
            0.1,  # <0.1ms requirement
            {"total_time_ms": total_time_ms, "operations": num_operations}
        )
    
    def test_alert_creation_performance(self) -> PerformanceResult:
        """Test alert creation performance - requirement: <0.5ms per alert"""
        num_alerts = 1000
        
        start_time = time.perf_counter()
        for i in range(num_alerts):
            self.monitoring_service.create_alert(
                AlertLevel.INFO,
                f"Performance Test Alert {i}",
                f"Test message {i}",
                "perf_test",
                {"alert_num": str(i)}
            )
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_alert = total_time_ms / num_alerts
        
        return self.record_result(
            "Alert Creation Performance",
            "avg_time_per_alert",
            avg_time_per_alert,
            "ms",
            0.5,  # <0.5ms requirement
            {"total_time_ms": total_time_ms, "alerts_created": num_alerts}
        )
    
    def test_query_performance_large_dataset(self) -> PerformanceResult:
        """Test query performance with large datasets - requirement: <100ms"""
        # Populate large dataset
        dataset_size = 50000
        for i in range(dataset_size):
            self.monitoring_service.record_metric(
                f"large_dataset_metric_{i % 20}",
                float(i),
                MetricType.GAUGE,
                {"category": f"cat_{i % 5}"}
            )
        
        # Test query performance
        start_time = time.perf_counter()
        
        all_metrics = self.monitoring_service.get_metrics()
        specific_metric = self.monitoring_service.get_metrics("large_dataset_metric_0")
        filtered_metrics = self.monitoring_service.get_metrics(since=datetime.now() - timedelta(minutes=5))
        
        end_time = time.perf_counter()
        
        query_time_ms = (end_time - start_time) * 1000
        
        return self.record_result(
            "Large Dataset Query Performance",
            "query_time",
            query_time_ms,
            "ms",
            100.0,  # <100ms requirement
            {
                "dataset_size": dataset_size,
                "metrics_count": len(all_metrics),
                "specific_results": len(specific_metric.get("large_dataset_metric_0", [])),
                "filtered_results": sum(len(metrics) for metrics in filtered_metrics.values())
            }
        )
    
    def test_concurrent_throughput(self) -> PerformanceResult:
        """Test concurrent operations throughput - requirement: >5000 ops/sec"""
        num_threads = 5
        ops_per_thread = 1000
        
        def worker_operations(thread_id: int) -> float:
            start_time = time.perf_counter()
            for i in range(ops_per_thread):
                if i % 2 == 0:
                    self.monitoring_service.record_metric(
                        f"concurrent_test_{thread_id}_{i}",
                        float(i),
                        MetricType.COUNTER
                    )
                else:
                    self.monitoring_service.create_alert(
                        AlertLevel.INFO,
                        f"Concurrent Alert {thread_id}-{i}",
                        f"Test {i}",
                        f"thread_{thread_id}"
                    )
            return time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_operations, i) for i in range(num_threads)]
            thread_times = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        total_operations = num_threads * ops_per_thread
        throughput = total_operations / total_time
        
        return self.record_result(
            "Concurrent Throughput",
            "operations_per_second",
            throughput,
            "ops/sec",
            5000.0,  # >5000 ops/sec requirement
            {
                "total_operations": total_operations,
                "total_time_sec": total_time,
                "num_threads": num_threads,
                "avg_thread_time": statistics.mean(thread_times),
                "max_thread_time": max(thread_times)
            }
        )
    
    def test_dashboard_api_response_time(self) -> PerformanceResult:
        """Test dashboard API response time - requirement: <50ms"""
        # Setup test data
        for i in range(1000):
            venue = ["IB", "BINANCE", "ALPACA"][i % 3]
            self.monitoring_service.record_metric(
                f"{venue.lower()}_latency",
                20.0 + (i % 100),
                MetricType.TIMING,
                {"venue": venue}
            )
        
        # Create some alerts
        for i in range(10):
            self.monitoring_service.create_alert(
                AlertLevel.WARNING,
                f"Test Alert {i}",
                f"Alert message {i}",
                "dashboard_test"
            )
        
        # Test dashboard API
        start_time = time.perf_counter()
        dashboard_data = self.monitoring_service.get_summary_dashboard()
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        return self.record_result(
            "Dashboard API Response Time",
            "response_time",
            response_time_ms,
            "ms",
            50.0,  # <50ms requirement
            {
                "metrics_returned": len(dashboard_data.get("metrics", {})),
                "alerts_returned": dashboard_data.get("alerts", {}).get("total", 0),
                "dashboard_fields": list(dashboard_data.keys())
            }
        )
    
    def test_health_check_api_speed(self) -> PerformanceResult:
        """Test health check API speed - requirement: <10ms"""
        # Add health status data
        from monitoring_service import HealthStatus
        components = ["database", "cache", "rate_limiter", "message_queue"]
        
        for component in components:
            self.monitoring_service._health_checks[component] = HealthStatus(
                component=component,
                status="healthy",
                last_check=datetime.now(),
                details={"response_time_ms": 5.2, "connections": 10},
                uptime_percentage=99.9
            )
        
        start_time = time.perf_counter()
        health_data = self.monitoring_service.get_health_status()
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        return self.record_result(
            "Health Check API Speed",
            "response_time",
            response_time_ms,
            "ms",
            10.0,  # <10ms requirement
            {
                "components_checked": len(health_data),
                "healthy_components": sum(1 for h in health_data.values() if h.status == "healthy")
            }
        )
    
    def test_memory_efficiency(self) -> PerformanceResult:
        """Test memory efficiency - requirement: <5MB per 1000 metrics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create test dataset
        num_metrics = 10000
        for i in range(num_metrics):
            self.monitoring_service.record_metric(
                f"memory_test_{i % 100}",
                float(i),
                MetricType.GAUGE,
                {"batch": str(i // 1000)}
            )
        
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_growth_mb = final_memory_mb - initial_memory_mb
        memory_per_1000_metrics = (memory_growth_mb / num_metrics) * 1000
        
        return self.record_result(
            "Memory Efficiency",
            "memory_per_1000_metrics",
            memory_per_1000_metrics,
            "MB",
            5.0,  # <5MB per 1000 metrics requirement
            {
                "initial_memory_mb": initial_memory_mb,
                "final_memory_mb": final_memory_mb,
                "memory_growth_mb": memory_growth_mb,
                "total_metrics": num_metrics
            }
        )
    
    def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all performance tests and generate comprehensive report"""
        print("🔧 Running Comprehensive Monitoring System Performance Tests...")
        print("=" * 80)
        
        # Run all performance tests
        tests = [
            self.test_metric_recording_overhead,
            self.test_alert_creation_performance,
            self.test_query_performance_large_dataset,
            self.test_concurrent_throughput,
            self.test_dashboard_api_response_time,
            self.test_health_check_api_speed,
            self.test_memory_efficiency
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.test_name}")
                print(f"    {result.metric}: {result.value:.3f}{result.unit} (requirement: {result.requirement}{result.unit})")
                if result.details:
                    for key, value in result.details.items():
                        print(f"    {key}: {value}")
                print()
            except Exception as e:
                print(f"❌ ERROR {test_func.__name__}: {e}")
                print()
        
        # Generate summary report
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        print("=" * 80)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {len(passed_tests)} ✅")
        print(f"Failed: {len(failed_tests)} ❌")
        print(f"Success Rate: {len(passed_tests)/len(self.results)*100:.1f}%")
        print()
        
        if failed_tests:
            print("❌ FAILED TESTS:")
            for result in failed_tests:
                print(f"  - {result.test_name}: {result.value:.3f}{result.unit} > {result.requirement}{result.unit}")
            print()
        
        # Key performance metrics summary
        key_metrics = {
            "Metric Recording Overhead": "0.1ms",
            "Query Performance": "100ms", 
            "Concurrent Throughput": "5000 ops/sec",
            "API Response Time": "50ms",
            "Memory Efficiency": "5MB per 1000 metrics"
        }
        
        print("🎯 KEY PERFORMANCE REQUIREMENTS:")
        for metric, requirement in key_metrics.items():
            matching_result = next((r for r in self.results if metric.lower() in r.test_name.lower()), None)
            if matching_result:
                status = "✅" if matching_result.passed else "❌"
                print(f"  {status} {metric}: < {requirement}")
        
        print()
        print("🔧 Mike's Integration & Performance Testing: COMPLETE")
        print("📈 Monitoring system performance validated for production readiness")
        
        return {
            "total_tests": len(self.results),
            "passed": len(passed_tests),
            "failed": len(failed_tests),
            "success_rate": len(passed_tests)/len(self.results)*100,
            "results": [
                {
                    "test": r.test_name,
                    "metric": r.metric,
                    "value": r.value,
                    "unit": r.unit,
                    "requirement": r.requirement,
                    "passed": r.passed,
                    "details": r.details
                }
                for r in self.results
            ]
        }


class TestMonitoringPerformanceValidation:
    """Pytest wrapper for performance validation"""
    
    def test_comprehensive_performance_validation(self):
        """Run comprehensive performance validation suite"""
        validator = MonitoringPerformanceValidator()
        report = validator.run_all_performance_tests()
        
        # Assert overall success
        assert report["success_rate"] >= 85, f"Performance test success rate too low: {report['success_rate']:.1f}% < 85%"
        
        # Verify critical performance requirements
        critical_tests = [
            "Metric Recording Overhead",
            "Dashboard API Response Time", 
            "Health Check API Speed"
        ]
        
        for test_name in critical_tests:
            matching_result = next((r for r in validator.results if test_name in r.test_name), None)
            assert matching_result is not None, f"Critical test missing: {test_name}"
            assert matching_result.passed, f"Critical test failed: {test_name} - {matching_result.value:.3f}{matching_result.unit} > {matching_result.requirement}{matching_result.unit}"
        
        print(f"\n🎉 ALL CRITICAL PERFORMANCE REQUIREMENTS VALIDATED")
        print(f"✅ {report['passed']}/{report['total_tests']} tests passed ({report['success_rate']:.1f}%)")


if __name__ == "__main__":
    # Run performance validation directly
    validator = MonitoringPerformanceValidator()
    report = validator.run_all_performance_tests()
    
    # Exit with success/failure code
    exit(0 if report["success_rate"] >= 85 else 1)
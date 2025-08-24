#!/usr/bin/env python3
"""
Test script for Real-Time Portfolio Analytics MessageBus Integration
Tests the enhanced risk engine with event-driven analytics processing
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Test configuration
TEST_PORTFOLIO_ID = "TEST_PORTFOLIO_001"
TEST_RISK_ENGINE_URL = "http://localhost:8200"

class RealTimeAnalyticsTest:
    """Test suite for real-time portfolio analytics"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
    def generate_test_data(self, days: int = 252) -> Dict[str, Any]:
        """Generate realistic test portfolio data"""
        np.random.seed(42)
        
        # Generate daily returns (realistic parameters)
        returns = np.random.normal(0.0008, 0.015, days)  # ~8% annual return, 15% volatility
        
        # Generate benchmark returns (slightly lower)
        benchmark_returns = np.random.normal(0.0006, 0.012, days)
        
        # Sample portfolio positions
        positions = {
            "AAPL": 0.25,
            "GOOGL": 0.20,
            "MSFT": 0.15,
            "AMZN": 0.15,
            "TSLA": 0.10,
            "NVDA": 0.10,
            "META": 0.05
        }
        
        return {
            "portfolio_id": TEST_PORTFOLIO_ID,
            "returns_history": returns.tolist(),
            "positions": positions,
            "benchmark_returns": benchmark_returns.tolist(),
            "computation_mode": "hybrid_auto",
            "priority": "high"
        }
    
    async def test_realtime_analytics_trigger(self) -> Dict[str, Any]:
        """Test triggering real-time analytics via MessageBus"""
        import aiohttp
        
        print("🧪 Testing Real-Time Analytics Trigger...")
        test_data = self.generate_test_data()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Trigger real-time analytics
                start_time = time.time()
                
                async with session.post(
                    f"{TEST_RISK_ENGINE_URL}/risk/analytics/realtime/trigger",
                    json=test_data
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "test": "realtime_analytics_trigger",
                            "status": "PASS",
                            "response_time_ms": response_time,
                            "queue_status": result.get("queue_status", {}),
                            "details": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "test": "realtime_analytics_trigger",
                            "status": "FAIL",
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time_ms": response_time
                        }
                        
        except Exception as e:
            return {
                "test": "realtime_analytics_trigger",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_realtime_optimization_trigger(self) -> Dict[str, Any]:
        """Test triggering real-time optimization via MessageBus"""
        import aiohttp
        
        print("🔧 Testing Real-Time Optimization Trigger...")
        
        test_data = {
            "portfolio_id": TEST_PORTFOLIO_ID,
            "assets": ["AAPL", "GOOGL", "MSFT", "AMZN"],
            "method": "minimum_variance",
            "constraints": {
                "min_weight": 0.0,
                "max_weight": 0.5
            },
            "priority": "high"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{TEST_RISK_ENGINE_URL}/risk/optimization/realtime/trigger",
                    json=test_data
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "test": "realtime_optimization_trigger",
                            "status": "PASS",
                            "response_time_ms": response_time,
                            "queue_status": result.get("queue_status", {}),
                            "details": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "test": "realtime_optimization_trigger",
                            "status": "FAIL",
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time_ms": response_time
                        }
                        
        except Exception as e:
            return {
                "test": "realtime_optimization_trigger",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_realtime_status_endpoint(self) -> Dict[str, Any]:
        """Test real-time analytics status endpoint"""
        import aiohttp
        
        print("📊 Testing Real-Time Analytics Status...")
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.get(
                    f"{TEST_RISK_ENGINE_URL}/risk/analytics/realtime/status"
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Validate key metrics
                        performance_targets = result.get("performance_targets", {})
                        worker_status = result.get("worker_status", {})
                        
                        validation_results = {
                            "worker_count_correct": worker_status.get("active_workers") == 8,
                            "processing_active": worker_status.get("processing_active", False),
                            "has_priority_queues": len(result.get("priority_queues", {})) == 4,
                            "has_performance_metrics": "event_processing_metrics" in result,
                            "hybrid_engine_healthy": result.get("hybrid_engine", {}).get("health") == "healthy"
                        }
                        
                        return {
                            "test": "realtime_status_endpoint",
                            "status": "PASS",
                            "response_time_ms": response_time,
                            "validation_results": validation_results,
                            "performance_targets": performance_targets,
                            "capabilities": result.get("capabilities", {}),
                            "details": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "test": "realtime_status_endpoint",
                            "status": "FAIL",
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time_ms": response_time
                        }
                        
        except Exception as e:
            return {
                "test": "realtime_status_endpoint",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_health_endpoint_enhancements(self) -> Dict[str, Any]:
        """Test that health endpoint includes new analytics metrics"""
        import aiohttp
        
        print("🏥 Testing Health Endpoint Enhancements...")
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.get(
                    f"{TEST_RISK_ENGINE_URL}/health"
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Check for new real-time analytics section
                        real_time_analytics = result.get("real_time_analytics", {})
                        
                        validation_results = {
                            "has_real_time_analytics": "real_time_analytics" in result,
                            "has_event_metrics": "portfolio_events_processed" in real_time_analytics,
                            "has_priority_queues": "priority_queues_status" in real_time_analytics,
                            "meets_performance_target": real_time_analytics.get("meets_performance_target", False),
                            "handles_target_throughput": real_time_analytics.get("handles_target_throughput", False)
                        }
                        
                        return {
                            "test": "health_endpoint_enhancements",
                            "status": "PASS",
                            "response_time_ms": response_time,
                            "validation_results": validation_results,
                            "real_time_analytics": real_time_analytics,
                            "details": result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "test": "health_endpoint_enhancements",
                            "status": "FAIL",
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time_ms": response_time
                        }
                        
        except Exception as e:
            return {
                "test": "health_endpoint_enhancements",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under multiple concurrent requests"""
        import aiohttp
        import asyncio
        
        print("⚡ Testing Performance Under Load...")
        
        num_concurrent_requests = 10
        test_data = self.generate_test_data(100)  # Smaller dataset for load testing
        
        async def make_request(session, request_id):
            """Make a single analytics request"""
            test_data_copy = test_data.copy()
            test_data_copy["portfolio_id"] = f"{TEST_PORTFOLIO_ID}_LOAD_{request_id}"
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{TEST_RISK_ENGINE_URL}/risk/analytics/realtime/trigger",
                    json=test_data_copy
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    success = response.status == 200
                    
                    return {
                        "request_id": request_id,
                        "success": success,
                        "response_time_ms": response_time,
                        "status_code": response.status
                    }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "error": str(e)
                }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Launch concurrent requests
                load_test_start = time.time()
                
                tasks = [
                    make_request(session, i) 
                    for i in range(num_concurrent_requests)
                ]
                
                results = await asyncio.gather(*tasks)
                
                total_test_time = (time.time() - load_test_start) * 1000
                
                # Analyze results
                successful_requests = sum(1 for r in results if r["success"])
                failed_requests = len(results) - successful_requests
                
                response_times = [r["response_time_ms"] for r in results if r["success"]]
                avg_response_time = np.mean(response_times) if response_times else 0
                max_response_time = np.max(response_times) if response_times else 0
                min_response_time = np.min(response_times) if response_times else 0
                
                requests_per_second = (num_concurrent_requests / total_test_time) * 1000
                
                return {
                    "test": "performance_under_load",
                    "status": "PASS" if successful_requests >= num_concurrent_requests * 0.8 else "FAIL",
                    "concurrent_requests": num_concurrent_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": (successful_requests / num_concurrent_requests) * 100,
                    "total_test_time_ms": total_test_time,
                    "requests_per_second": requests_per_second,
                    "response_time_stats": {
                        "avg_ms": avg_response_time,
                        "max_ms": max_response_time,
                        "min_ms": min_response_time
                    },
                    "meets_performance_targets": {
                        "avg_under_50ms": avg_response_time < 50,
                        "max_under_100ms": max_response_time < 100,
                        "success_rate_over_90pct": (successful_requests / num_concurrent_requests) >= 0.9
                    },
                    "details": results
                }
                
        except Exception as e:
            return {
                "test": "performance_under_load",
                "status": "ERROR",
                "error": str(e)
            }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all tests and return results"""
        print("🚀 Starting Real-Time Portfolio Analytics Test Suite...")
        print("=" * 80)
        
        tests = [
            self.test_health_endpoint_enhancements(),
            self.test_realtime_status_endpoint(),
            self.test_realtime_analytics_trigger(),
            self.test_realtime_optimization_trigger(),
            self.test_performance_under_load()
        ]
        
        results = []
        for i, test in enumerate(tests, 1):
            print(f"\n[{i}/{len(tests)}] Running test...")
            result = await test
            results.append(result)
            
            # Print test result
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            print(f"{status_icon} {result['test']}: {result['status']}")
            
            if result["status"] != "PASS":
                print(f"    Error: {result.get('error', 'Unknown error')}")
            
            if "response_time_ms" in result:
                print(f"    Response time: {result['response_time_ms']:.1f}ms")
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📋 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["status"] == "PASS")
        failed_tests = sum(1 for r in results if r["status"] == "FAIL")
        error_tests = sum(1 for r in results if r["status"] == "ERROR")
        
        total_time = time.time() - self.start_time
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Errors: {error_tests}")
        print(f"🕐 Total Time: {total_time:.1f}s")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        
        for result in results:
            if "response_time_ms" in result:
                print(f"{result['test']}: {result['response_time_ms']:.1f}ms")
        
        # Check if performance targets met
        performance_result = next((r for r in results if r["test"] == "performance_under_load"), None)
        if performance_result and performance_result["status"] == "PASS":
            targets = performance_result.get("meets_performance_targets", {})
            print(f"\n🎯 PERFORMANCE TARGETS")
            print("-" * 40)
            for target, met in targets.items():
                icon = "✅" if met else "❌"
                print(f"{icon} {target.replace('_', ' ').title()}")
        
        print("\n" + "=" * 80)
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Real-time analytics integration is working correctly.")
        else:
            print(f"⚠️  {failed_tests + error_tests} test(s) failed. Please check the implementation.")
        
        print("=" * 80)

async def main():
    """Main test execution"""
    try:
        print("🧪 Real-Time Portfolio Analytics MessageBus Integration Test")
        print(f"Testing against: {TEST_RISK_ENGINE_URL}")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_suite = RealTimeAnalyticsTest()
        results = await test_suite.run_all_tests()
        test_suite.print_summary(results)
        
        # Save detailed results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"realtime_analytics_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "test_configuration": {
                    "portfolio_id": TEST_PORTFOLIO_ID,
                    "risk_engine_url": TEST_RISK_ENGINE_URL,
                    "concurrent_requests": 10
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import aiohttp
        import numpy
        import pandas
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install aiohttp numpy pandas")
        exit(1)
    
    asyncio.run(main())
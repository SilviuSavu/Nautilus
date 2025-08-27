#!/usr/bin/env python3
"""
🚀 MarketData Hub Load Testing & Stress Testing
Quinn's Performance Validation Under High Load Conditions

Tests MarketData Hub and Client performance under various load scenarios:
- Concurrent request handling
- High frequency data requests  
- Memory usage under load
- Error rate analysis
- Cache performance validation
- Throughput measurement
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add backend to path
sys.path.append('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend')

from marketdata_client import MarketDataClient, DataType, DataSource
from universal_enhanced_messagebus_client import EngineType, MessagePriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataLoadTester:
    """Comprehensive load testing for MarketData Hub"""
    
    def __init__(self, hub_url: str = "http://localhost:8800"):
        self.hub_url = hub_url
        self.results = []
        self.errors = []
        
    async def test_concurrent_http_requests(self, num_requests: int = 100, concurrency: int = 10) -> Dict[str, Any]:
        """Test concurrent HTTP requests to MarketData Hub"""
        
        print(f"🔥 Testing {num_requests} concurrent HTTP requests (concurrency: {concurrency})...")
        
        async def single_request(session: aiohttp.ClientSession, request_id: int) -> Tuple[float, int, Dict]:
            """Single HTTP request with timing"""
            start_time = time.time()
            
            try:
                async with session.post(
                    f"{self.hub_url}/data/request",
                    json={
                        "symbols": ["AAPL", "GOOGL"],
                        "data_types": ["quote", "level2"],
                        "sources": ["ibkr"],
                        "engine": "load_test",
                        "cache": True
                    }
                ) as response:
                    result = await response.json()
                    elapsed = (time.time() - start_time) * 1000
                    return elapsed, response.status, result
                    
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                self.errors.append(f"Request {request_id}: {str(e)}")
                return elapsed, 500, {"error": str(e)}
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(session: aiohttp.ClientSession, request_id: int):
            async with semaphore:
                return await single_request(session, request_id)
        
        # Execute concurrent requests
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [limited_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        response_times = [r[0] for r in results]
        status_codes = [r[1] for r in results]
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        
        return {
            "test_type": "concurrent_http",
            "num_requests": num_requests,
            "concurrency": concurrency,
            "total_time_seconds": total_time,
            "requests_per_second": num_requests / total_time,
            "successful_requests": successful_requests,
            "error_rate": (num_requests - successful_requests) / num_requests,
            "response_times": {
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "avg_ms": statistics.mean(response_times),
                "median_ms": statistics.median(response_times),
                "p95_ms": self._percentile(response_times, 95),
                "p99_ms": self._percentile(response_times, 99)
            }
        }
    
    async def test_high_frequency_requests(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Test high frequency requests for sustained period"""
        
        print(f"⚡ Testing high frequency requests for {duration_seconds} seconds...")
        
        request_count = 0
        response_times = []
        errors = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                request_start = time.time()
                
                try:
                    async with session.post(
                        f"{self.hub_url}/data/request",
                        json={
                            "symbols": ["AAPL"],
                            "data_types": ["quote"],
                            "sources": ["ibkr"],
                            "engine": "frequency_test",
                            "cache": True
                        }
                    ) as response:
                        if response.status == 200:
                            await response.json()
                        request_count += 1
                        
                except Exception as e:
                    errors += 1
                
                response_time = (time.time() - request_start) * 1000
                response_times.append(response_time)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
        
        total_time = time.time() - start_time
        
        return {
            "test_type": "high_frequency",
            "duration_seconds": total_time,
            "total_requests": request_count,
            "requests_per_second": request_count / total_time,
            "errors": errors,
            "error_rate": errors / max(1, request_count + errors),
            "response_times": {
                "avg_ms": statistics.mean(response_times) if response_times else 0,
                "median_ms": statistics.median(response_times) if response_times else 0,
                "max_ms": max(response_times) if response_times else 0,
                "min_ms": min(response_times) if response_times else 0
            }
        }
    
    async def test_cache_performance_under_load(self, num_requests: int = 200) -> Dict[str, Any]:
        """Test cache hit rate and performance under load"""
        
        print(f"💾 Testing cache performance with {num_requests} requests...")
        
        # Phase 1: Prime cache with different symbols
        prime_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "META"]
        
        async with aiohttp.ClientSession() as session:
            # Prime cache
            for symbol in prime_symbols:
                await session.post(
                    f"{self.hub_url}/data/request",
                    json={
                        "symbols": [symbol],
                        "data_types": ["quote"],
                        "sources": ["ibkr"],
                        "engine": "cache_test",
                        "cache": True
                    }
                )
            
            # Phase 2: Mixed cache hits and misses
            cache_hit_times = []
            cache_miss_times = []
            
            for i in range(num_requests):
                # Alternate between cached symbols and new symbols
                if i % 2 == 0:
                    # Cache hit - use primed symbol
                    symbol = prime_symbols[i % len(prime_symbols)]
                    is_cached = True
                else:
                    # Cache miss - use unique symbol
                    symbol = f"TEST{i}"
                    is_cached = False
                
                start_time = time.time()
                
                async with session.post(
                    f"{self.hub_url}/data/request",
                    json={
                        "symbols": [symbol],
                        "data_types": ["quote"],
                        "sources": ["ibkr"],
                        "engine": "cache_test",
                        "cache": True
                    }
                ) as response:
                    await response.json()
                    
                response_time = (time.time() - start_time) * 1000
                
                if is_cached:
                    cache_hit_times.append(response_time)
                else:
                    cache_miss_times.append(response_time)
        
        # Calculate cache performance improvement
        avg_hit_time = statistics.mean(cache_hit_times) if cache_hit_times else 0
        avg_miss_time = statistics.mean(cache_miss_times) if cache_miss_times else 0
        cache_improvement = avg_miss_time / max(0.1, avg_hit_time)
        
        return {
            "test_type": "cache_performance",
            "total_requests": num_requests,
            "cache_hits": len(cache_hit_times),
            "cache_misses": len(cache_miss_times),
            "cache_hit_rate": len(cache_hit_times) / num_requests,
            "cache_hit_avg_ms": avg_hit_time,
            "cache_miss_avg_ms": avg_miss_time,
            "cache_performance_improvement": cache_improvement
        }
    
    async def test_memory_usage_under_load(self, num_requests: int = 500) -> Dict[str, Any]:
        """Test memory usage patterns under sustained load"""
        
        print(f"🧠 Testing memory usage with {num_requests} requests...")
        
        # Get initial memory stats if available
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hub_url}/metrics") as response:
                    initial_metrics = await response.json()
                    initial_cache_size = initial_metrics.get("cache_metrics", {}).get("cache_size", 0)
        except:
            initial_cache_size = 0
        
        # Generate load with many different symbols
        symbols = [f"LOAD_TEST_{i}" for i in range(num_requests)]
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Send requests in batches to avoid overwhelming
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                tasks = []
                for symbol in batch:
                    task = session.post(
                        f"{self.hub_url}/data/request",
                        json={
                            "symbols": [symbol],
                            "data_types": ["quote"],
                            "sources": ["ibkr"],
                            "engine": "memory_test",
                            "cache": True
                        }
                    )
                    tasks.append(task)
                
                # Execute batch
                responses = await asyncio.gather(*tasks)
                for response in responses:
                    response.close()
                
                # Small delay between batches
                await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # Get final memory stats
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hub_url}/metrics") as response:
                    final_metrics = await response.json()
                    final_cache_size = final_metrics.get("cache_metrics", {}).get("cache_size", 0)
        except:
            final_cache_size = 0
        
        return {
            "test_type": "memory_usage",
            "total_requests": num_requests,
            "duration_seconds": total_time,
            "initial_cache_size": initial_cache_size,
            "final_cache_size": final_cache_size,
            "cache_growth": final_cache_size - initial_cache_size,
            "requests_per_second": num_requests / total_time
        }
    
    async def test_error_handling_under_load(self) -> Dict[str, Any]:
        """Test error handling and recovery under load"""
        
        print("💥 Testing error handling under load...")
        
        total_requests = 0
        successful_requests = 0
        errors_by_type = {}
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Invalid symbols
            for i in range(50):
                total_requests += 1
                try:
                    async with session.post(
                        f"{self.hub_url}/data/request",
                        json={
                            "symbols": ["INVALID_SYMBOL_" + str(i)],
                            "data_types": ["quote"],
                            "sources": ["ibkr"],
                            "engine": "error_test",
                            "cache": True
                        }
                    ) as response:
                        if response.status == 200:
                            successful_requests += 1
                        else:
                            error_type = f"HTTP_{response.status}"
                            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
                except Exception as e:
                    error_type = type(e).__name__
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            
            # Test 2: Invalid data types
            for i in range(25):
                total_requests += 1
                try:
                    async with session.post(
                        f"{self.hub_url}/data/request",
                        json={
                            "symbols": ["AAPL"],
                            "data_types": ["invalid_data_type"],
                            "sources": ["ibkr"],
                            "engine": "error_test",
                            "cache": True
                        }
                    ) as response:
                        if response.status == 200:
                            successful_requests += 1
                        else:
                            error_type = f"HTTP_{response.status}"
                            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
                except Exception as e:
                    error_type = type(e).__name__
                    errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            
            # Test 3: Large payloads
            large_symbols = [f"SYM_{i}" for i in range(100)]
            total_requests += 1
            try:
                async with session.post(
                    f"{self.hub_url}/data/request",
                    json={
                        "symbols": large_symbols,
                        "data_types": ["quote", "level2", "fundamental"],
                        "sources": ["ibkr", "alpha_vantage"],
                        "engine": "error_test",
                        "cache": True
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        successful_requests += 1
                    else:
                        error_type = f"HTTP_{response.status}"
                        errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            except Exception as e:
                error_type = type(e).__name__
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        return {
            "test_type": "error_handling",
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate": (total_requests - successful_requests) / total_requests,
            "errors_by_type": errors_by_type
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def run_comprehensive_load_tests(self) -> Dict[str, Any]:
        """Run all load tests and return comprehensive results"""
        
        print("🚀 COMPREHENSIVE MARKETDATA LOAD TESTING SUITE")
        print("=" * 60)
        print(f"🕐 Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target Hub: {self.hub_url}")
        print()
        
        results = {}
        
        try:
            # Test 1: Concurrent HTTP requests
            print("Test 1/5: Concurrent HTTP Requests")
            results["concurrent_http_light"] = await self.test_concurrent_http_requests(50, 5)
            results["concurrent_http_heavy"] = await self.test_concurrent_http_requests(200, 20)
            
            # Test 2: High frequency requests  
            print("\nTest 2/5: High Frequency Requests")
            results["high_frequency"] = await self.test_high_frequency_requests(15)
            
            # Test 3: Cache performance
            print("\nTest 3/5: Cache Performance Under Load")
            results["cache_performance"] = await self.test_cache_performance_under_load(100)
            
            # Test 4: Memory usage
            print("\nTest 4/5: Memory Usage Under Load")
            results["memory_usage"] = await self.test_memory_usage_under_load(300)
            
            # Test 5: Error handling
            print("\nTest 5/5: Error Handling Under Load")
            results["error_handling"] = await self.test_error_handling_under_load()
            
        except Exception as e:
            logger.error(f"Load testing failed: {e}")
            results["error"] = str(e)
        
        return results

# Performance analysis and reporting
def analyze_performance_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance results and provide insights"""
    
    analysis = {
        "performance_grade": "UNKNOWN",
        "bottlenecks": [],
        "recommendations": [],
        "sla_compliance": {}
    }
    
    # SLA targets
    SLA_TARGETS = {
        "max_response_time_ms": 50,  # 50ms max response time
        "min_requests_per_second": 100,  # 100 RPS minimum
        "max_error_rate": 0.05,  # 5% error rate maximum
        "min_cache_hit_rate": 0.80,  # 80% cache hit rate minimum
        "cache_improvement_factor": 3.0  # 3x improvement from cache
    }
    
    # Analyze concurrent requests
    if "concurrent_http_heavy" in results:
        heavy_load = results["concurrent_http_heavy"]
        
        # Response time analysis
        p95_response = heavy_load["response_times"]["p95_ms"]
        if p95_response <= SLA_TARGETS["max_response_time_ms"]:
            analysis["sla_compliance"]["response_time"] = "PASS"
        else:
            analysis["sla_compliance"]["response_time"] = "FAIL"
            analysis["bottlenecks"].append(f"P95 response time {p95_response:.1f}ms exceeds {SLA_TARGETS['max_response_time_ms']}ms target")
        
        # Throughput analysis
        rps = heavy_load["requests_per_second"]
        if rps >= SLA_TARGETS["min_requests_per_second"]:
            analysis["sla_compliance"]["throughput"] = "PASS"
        else:
            analysis["sla_compliance"]["throughput"] = "FAIL"
            analysis["bottlenecks"].append(f"Throughput {rps:.1f} RPS below {SLA_TARGETS['min_requests_per_second']} RPS target")
        
        # Error rate analysis
        error_rate = heavy_load["error_rate"]
        if error_rate <= SLA_TARGETS["max_error_rate"]:
            analysis["sla_compliance"]["error_rate"] = "PASS"
        else:
            analysis["sla_compliance"]["error_rate"] = "FAIL"
            analysis["bottlenecks"].append(f"Error rate {error_rate:.1%} exceeds {SLA_TARGETS['max_error_rate']:.1%} target")
    
    # Analyze cache performance
    if "cache_performance" in results:
        cache_perf = results["cache_performance"]
        
        hit_rate = cache_perf["cache_hit_rate"]
        if hit_rate >= SLA_TARGETS["min_cache_hit_rate"]:
            analysis["sla_compliance"]["cache_hit_rate"] = "PASS"
        else:
            analysis["sla_compliance"]["cache_hit_rate"] = "FAIL"
            analysis["bottlenecks"].append(f"Cache hit rate {hit_rate:.1%} below {SLA_TARGETS['min_cache_hit_rate']:.1%} target")
        
        improvement = cache_perf["cache_performance_improvement"]
        if improvement >= SLA_TARGETS["cache_improvement_factor"]:
            analysis["sla_compliance"]["cache_improvement"] = "PASS"
        else:
            analysis["sla_compliance"]["cache_improvement"] = "FAIL"
            analysis["bottlenecks"].append(f"Cache improvement {improvement:.1f}x below {SLA_TARGETS['cache_improvement_factor']:.1f}x target")
    
    # Generate overall grade
    passing_slas = sum(1 for status in analysis["sla_compliance"].values() if status == "PASS")
    total_slas = len(analysis["sla_compliance"])
    
    if passing_slas == total_slas:
        analysis["performance_grade"] = "A+ EXCELLENT"
    elif passing_slas >= total_slas * 0.8:
        analysis["performance_grade"] = "B+ GOOD"  
    elif passing_slas >= total_slas * 0.6:
        analysis["performance_grade"] = "C ACCEPTABLE"
    else:
        analysis["performance_grade"] = "D NEEDS IMPROVEMENT"
    
    # Generate recommendations
    if analysis["bottlenecks"]:
        analysis["recommendations"].extend([
            "Consider increasing cache size and TTL values",
            "Implement connection pooling for better concurrency",
            "Add rate limiting to prevent overload",
            "Monitor memory usage and implement garbage collection",
            "Consider database query optimization"
        ])
    else:
        analysis["recommendations"].append("System performing within all SLA targets")
    
    return analysis

async def main():
    """Main load testing runner"""
    
    # Check if MarketData Hub is running
    tester = MarketDataLoadTester()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{tester.hub_url}/health") as response:
                if response.status != 200:
                    print(f"❌ MarketData Hub not healthy at {tester.hub_url}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to MarketData Hub at {tester.hub_url}: {e}")
        print("💡 Make sure the Centralized MarketData Hub is running on port 8800")
        return False
    
    # Run comprehensive load tests
    results = await tester.run_comprehensive_load_tests()
    
    # Analyze results
    analysis = analyze_performance_results(results)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 LOAD TESTING RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Overall Performance Grade: {analysis['performance_grade']}")
    
    print(f"\n✅ SLA Compliance:")
    for sla, status in analysis["sla_compliance"].items():
        indicator = "✅" if status == "PASS" else "❌"
        print(f"   {indicator} {sla}: {status}")
    
    if analysis["bottlenecks"]:
        print(f"\n⚠️ Performance Bottlenecks:")
        for bottleneck in analysis["bottlenecks"]:
            print(f"   • {bottleneck}")
    
    print(f"\n💡 Recommendations:")
    for rec in analysis["recommendations"]:
        print(f"   • {rec}")
    
    # Detailed metrics
    print(f"\n📈 Detailed Performance Metrics:")
    
    if "concurrent_http_heavy" in results:
        heavy = results["concurrent_http_heavy"]
        print(f"   Concurrent Load (200 requests, 20 concurrency):")
        print(f"     • Throughput: {heavy['requests_per_second']:.1f} RPS")
        print(f"     • P95 Response Time: {heavy['response_times']['p95_ms']:.1f}ms")
        print(f"     • Error Rate: {heavy['error_rate']:.1%}")
    
    if "high_frequency" in results:
        freq = results["high_frequency"]
        print(f"   High Frequency Test:")
        print(f"     • Sustained RPS: {freq['requests_per_second']:.1f}")
        print(f"     • Average Response: {freq['response_times']['avg_ms']:.1f}ms")
    
    if "cache_performance" in results:
        cache = results["cache_performance"]
        print(f"   Cache Performance:")
        print(f"     • Hit Rate: {cache['cache_hit_rate']:.1%}")
        print(f"     • Performance Improvement: {cache['cache_performance_improvement']:.1f}x")
    
    print(f"\n🕐 Test End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save detailed results
    with open("marketdata_load_test_results.json", "w") as f:
        json.dump({
            "results": results,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: marketdata_load_test_results.json")
    
    return analysis["performance_grade"].startswith("A") or analysis["performance_grade"].startswith("B")

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
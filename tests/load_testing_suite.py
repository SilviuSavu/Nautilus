#!/usr/bin/env python3
"""
Nautilus Load Testing Suite
Comprehensive load testing for all operational engines
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    engine: str
    test_type: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    requests_per_second: float
    success_rate: float
    duration_seconds: float

class LoadTestSuite:
    def __init__(self):
        self.operational_engines = {
            'analytics': {'port': 8100, 'name': 'Analytics Engine'},
            'ml': {'port': 8400, 'name': 'ML Engine'},
            'features': {'port': 8500, 'name': 'Features Engine'},
            'websocket': {'port': 8600, 'name': 'WebSocket Engine'},
            'strategy': {'port': 8700, 'name': 'Strategy Engine'},
            'marketdata': {'port': 8800, 'name': 'MarketData Engine'},
            'portfolio': {'port': 8900, 'name': 'Portfolio Engine'}
        }
        
        self.test_results = []
        
        # Test data
        self.test_portfolio_data = {
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "cost_basis": 150.0},
                {"symbol": "MSFT", "quantity": 50, "cost_basis": 300.0},
                {"symbol": "GOOGL", "quantity": 25, "cost_basis": 2800.0}
            ],
            "cash": 10000.0,
            "benchmark": "SPY"
        }
        
        self.test_market_data = [
            {"symbol": "AAPL", "price": 175.50, "timestamp": "2025-08-24T13:50:00Z"},
            {"symbol": "MSFT", "price": 320.25, "timestamp": "2025-08-24T13:50:00Z"},
            {"symbol": "GOOGL", "price": 2950.75, "timestamp": "2025-08-24T13:50:00Z"}
        ]
    
    async def load_test_engine(self, session: aiohttp.ClientSession, engine_id: str, 
                              concurrent_users: int = 10, requests_per_user: int = 50) -> LoadTestResult:
        """Perform load test on a specific engine"""
        engine_config = self.operational_engines[engine_id]
        port = engine_config['port']
        engine_name = engine_config['name']
        
        logger.info(f"Starting load test: {engine_name} ({concurrent_users} users, {requests_per_user} requests each)")
        
        async def user_session(user_id: int) -> List[float]:
            """Simulate a single user's requests"""
            response_times = []
            
            for request_num in range(requests_per_user):
                start_time = time.time()
                try:
                    # Choose appropriate endpoint based on engine
                    endpoint_url = self._get_test_endpoint(port, engine_id, request_num)
                    
                    async with session.get(endpoint_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        await response.text()  # Consume response
                        end_time = time.time()
                        
                        if response.status == 200:
                            response_time_ms = (end_time - start_time) * 1000
                            response_times.append(response_time_ms)
                        else:
                            response_times.append(-1)  # Mark as failed
                            
                except Exception as e:
                    response_times.append(-1)  # Mark as failed
                    
                # Small delay between requests
                await asyncio.sleep(0.01)
            
            return response_times
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [user_session(i) for i in range(concurrent_users)]
        all_response_times = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Flatten results
        flat_response_times = [rt for user_times in all_response_times for rt in user_times]
        
        # Calculate statistics
        successful_times = [rt for rt in flat_response_times if rt > 0]
        failed_count = len([rt for rt in flat_response_times if rt < 0])
        
        total_requests = len(flat_response_times)
        successful_requests = len(successful_times)
        duration_seconds = end_time - start_time
        
        if successful_times:
            avg_response_time = statistics.mean(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            p95_response_time = np.percentile(successful_times, 95)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        result = LoadTestResult(
            engine=engine_name,
            test_type=f"load_test_{concurrent_users}users_{requests_per_user}req",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_count,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p95_response_time_ms=p95_response_time,
            requests_per_second=total_requests / duration_seconds if duration_seconds > 0 else 0,
            success_rate=(successful_requests / total_requests * 100) if total_requests > 0 else 0,
            duration_seconds=duration_seconds
        )
        
        self.test_results.append(result)
        logger.info(f"Completed {engine_name}: {result.success_rate:.1f}% success, {result.avg_response_time_ms:.1f}ms avg")
        
        return result
    
    def _get_test_endpoint(self, port: int, engine_id: str, request_num: int) -> str:
        """Get appropriate test endpoint for each engine type"""
        base_url = f"http://localhost:{port}"
        
        # Vary endpoints to test different functionality
        endpoint_patterns = {
            'analytics': ['/health', '/metrics'],
            'ml': ['/health', '/metrics'],
            'features': ['/health', '/metrics'],
            'websocket': ['/health', '/metrics'],
            'strategy': ['/health', '/metrics'],
            'marketdata': ['/health', '/metrics'],
            'portfolio': ['/health', '/metrics']
        }
        
        patterns = endpoint_patterns.get(engine_id, ['/health'])
        endpoint = patterns[request_num % len(patterns)]
        
        return f"{base_url}{endpoint}"
    
    async def run_stress_test(self) -> List[LoadTestResult]:
        """Run comprehensive stress tests on all operational engines"""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE LOAD TESTING SUITE")
        logger.info("=" * 80)
        
        # Test configurations (users, requests per user)
        test_configs = [
            (5, 20),    # Light load
            (10, 50),   # Medium load
            (20, 25),   # Heavy load (more users, fewer requests each)
        ]
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30),
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            
            for users, requests in test_configs:
                logger.info(f"\n--- Running Test Configuration: {users} users, {requests} requests each ---")
                
                # Test all operational engines
                for engine_id in self.operational_engines.keys():
                    try:
                        await self.load_test_engine(session, engine_id, users, requests)
                    except Exception as e:
                        logger.error(f"Failed to test {engine_id}: {e}")
                
                # Brief pause between test configurations
                await asyncio.sleep(2)
        
        return self.test_results
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Group results by engine
        engine_results = {}
        for result in self.test_results:
            if result.engine not in engine_results:
                engine_results[result.engine] = []
            engine_results[result.engine].append(result)
        
        # Calculate aggregate metrics
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "engines_tested": len(engine_results),
                "test_timestamp": datetime.now().isoformat()
            },
            "overall_metrics": {
                "avg_success_rate": statistics.mean([r.success_rate for r in self.test_results]),
                "avg_response_time": statistics.mean([r.avg_response_time_ms for r in self.test_results if r.avg_response_time_ms > 0]),
                "total_requests_processed": sum([r.total_requests for r in self.test_results]),
                "total_successful_requests": sum([r.successful_requests for r in self.test_results])
            },
            "engine_performance": {}
        }
        
        # Detailed engine analysis
        for engine_name, results in engine_results.items():
            best_performance = min(results, key=lambda r: r.avg_response_time_ms if r.avg_response_time_ms > 0 else float('inf'))
            worst_performance = max(results, key=lambda r: r.avg_response_time_ms)
            
            engine_summary = {
                "tests_completed": len(results),
                "best_avg_response_time_ms": best_performance.avg_response_time_ms,
                "worst_avg_response_time_ms": worst_performance.avg_response_time_ms,
                "average_success_rate": statistics.mean([r.success_rate for r in results]),
                "max_requests_per_second": max([r.requests_per_second for r in results]),
                "total_requests_processed": sum([r.total_requests for r in results]),
                "grade": self._calculate_engine_grade(results)
            }
            
            report["engine_performance"][engine_name] = engine_summary
        
        # Performance recommendations
        report["recommendations"] = self._generate_recommendations(engine_results)
        
        return report
    
    def _calculate_engine_grade(self, results: List[LoadTestResult]) -> str:
        """Calculate performance grade for an engine"""
        avg_success_rate = statistics.mean([r.success_rate for r in results])
        avg_response_time = statistics.mean([r.avg_response_time_ms for r in results if r.avg_response_time_ms > 0])
        
        # Grading criteria
        if avg_success_rate >= 95 and avg_response_time <= 50:
            return "A+"
        elif avg_success_rate >= 90 and avg_response_time <= 100:
            return "A"
        elif avg_success_rate >= 85 and avg_response_time <= 200:
            return "B+"
        elif avg_success_rate >= 80 and avg_response_time <= 500:
            return "B"
        elif avg_success_rate >= 70:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(self, engine_results: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for engine_name, results in engine_results.items():
            avg_success_rate = statistics.mean([r.success_rate for r in results])
            avg_response_time = statistics.mean([r.avg_response_time_ms for r in results if r.avg_response_time_ms > 0])
            
            if avg_success_rate < 90:
                recommendations.append(f"{engine_name}: Improve reliability (success rate: {avg_success_rate:.1f}%)")
            
            if avg_response_time > 200:
                recommendations.append(f"{engine_name}: Optimize response time (avg: {avg_response_time:.1f}ms)")
            
            max_rps = max([r.requests_per_second for r in results])
            if max_rps < 50:
                recommendations.append(f"{engine_name}: Consider scaling for higher throughput (max: {max_rps:.1f} RPS)")
        
        if not recommendations:
            recommendations.append("Excellent performance across all engines!")
        
        return recommendations

async def main():
    """Run the comprehensive load testing suite"""
    load_tester = LoadTestSuite()
    
    # Run stress tests
    results = await load_tester.run_stress_test()
    
    # Generate and display report
    report = load_tester.generate_performance_report()
    
    logger.info("\n" + "=" * 80)
    logger.info("LOAD TESTING RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # Display summary
    summary = report["overall_metrics"]
    logger.info(f"Total Requests Processed: {summary['total_requests_processed']:,}")
    logger.info(f"Overall Success Rate: {summary['avg_success_rate']:.1f}%")
    logger.info(f"Average Response Time: {summary['avg_response_time']:.1f}ms")
    
    # Display engine grades
    logger.info("\nEngine Performance Grades:")
    for engine, perf in report["engine_performance"].items():
        logger.info(f"  {engine}: {perf['grade']} ({perf['average_success_rate']:.1f}% success, {perf['max_requests_per_second']:.1f} RPS)")
    
    # Display recommendations
    logger.info("\nRecommendations:")
    for rec in report["recommendations"]:
        logger.info(f"  • {rec}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {filename}")
    logger.info("Load testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
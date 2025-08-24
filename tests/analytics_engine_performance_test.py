#!/usr/bin/env python3
"""
Analytics Engine Comprehensive Performance Test
Tests the containerized Analytics Engine performance, capabilities, and integration
"""
import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
import numpy as np
import psutil
import docker
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalyticsEnginePerformanceTest:
    """
    Comprehensive performance testing suite for Analytics Engine
    Tests containerized performance, scalability, and integration capabilities
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8100"
        self.docker_client = docker.from_env()
        self.container_name = "nautilus-analytics-engine"
        self.test_data = {}
        self.performance_metrics = {}
        self.load_test_data()
        
    def load_test_data(self):
        """Load comprehensive test data for performance testing"""
        logger.info("Loading test data for Analytics Engine performance testing...")
        
        # Load historical price data for performance calculations
        try:
            # Load AAPL daily data
            with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_AAPL_daily.json', 'r') as f:
                aapl_daily = json.load(f)
                self.test_data['aapl_daily'] = aapl_daily
            
            # Load MSFT daily data
            with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_MSFT_daily.json', 'r') as f:
                msft_daily = json.load(f)
                self.test_data['msft_daily'] = msft_daily
                
            # Load synthetic high-frequency data
            with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/synthetic_AAPL_hf.json', 'r') as f:
                aapl_hf = json.load(f)
                self.test_data['aapl_hf'] = aapl_hf
                
            # Load portfolio data (multiple symbols)
            portfolio_data = {}
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            for symbol in symbols:
                try:
                    with open(f'/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_{symbol}_daily.json', 'r') as f:
                        portfolio_data[symbol] = json.load(f)
                except FileNotFoundError:
                    logger.warning(f"Could not load data for {symbol}")
                    
            self.test_data['portfolio'] = portfolio_data
            logger.info(f"Loaded test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            # Generate synthetic data if files not available
            self.generate_synthetic_test_data()
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data...")
        
        # Generate synthetic portfolio data
        np.random.seed(42)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        portfolio_data = {}
        for symbol in symbols:
            # Generate 2 years of daily data
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
            
            # Generate realistic price movements
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = prices[1:]  # Remove initial base_price
            
            # Create OHLCV data
            daily_data = {}
            for i, date in enumerate(dates):
                high = prices[i] * (1 + abs(np.random.normal(0, 0.005)))
                low = prices[i] * (1 - abs(np.random.normal(0, 0.005)))
                volume = int(np.random.exponential(1000000))
                
                daily_data[date.isoformat()] = {
                    'Open': round(prices[i] * (1 + np.random.normal(0, 0.001)), 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(prices[i], 2),
                    'Volume': volume,
                    'Adj Close': round(prices[i], 2)
                }
            
            portfolio_data[symbol] = daily_data
        
        self.test_data['portfolio'] = portfolio_data
        logger.info("Synthetic test data generated successfully")
    
    async def check_container_health(self) -> Dict[str, Any]:
        """Check if the Analytics Engine container is healthy"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate resource usage
            cpu_percent = 0.0
            memory_usage_mb = 0.0
            memory_limit_mb = 0.0
            
            if 'cpu_stats' in stats and 'precpu_stats' in stats:
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [0])) * 100
            
            if 'memory_stats' in stats:
                memory_usage_mb = stats['memory_stats'].get('usage', 0) / 1024 / 1024
                memory_limit_mb = stats['memory_stats'].get('limit', 0) / 1024 / 1024
            
            return {
                "container_id": container.id[:12],
                "status": container.status,
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage_mb, 2),
                "memory_limit_mb": round(memory_limit_mb, 2),
                "memory_percent": round((memory_usage_mb / memory_limit_mb) * 100, 2) if memory_limit_mb > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error checking container health: {e}")
            return {"error": str(e)}
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint and measure response time"""
        logger.info("Testing Analytics Engine health endpoint...")
        
        results = {
            "test_name": "health_endpoint",
            "test_timestamp": datetime.now().isoformat(),
            "response_times": [],
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "container_health": {}
        }
        
        # Test health endpoint multiple times
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}/health", timeout=5) as response:
                        response_time = (time.time() - start_time) * 1000  # Convert to ms
                        results["response_times"].append(response_time)
                        
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Health check {i+1}: {response_time:.2f}ms - {data.get('status', 'unknown')}")
                        else:
                            logger.warning(f"Health check {i+1}: HTTP {response.status}")
                
                except Exception as e:
                    logger.error(f"Health check {i+1} failed: {e}")
                
                await asyncio.sleep(0.1)  # Brief delay between requests
        
        # Calculate statistics
        if results["response_times"]:
            results["success_rate"] = len(results["response_times"]) / 10 * 100
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["p95_response_time"] = np.percentile(results["response_times"], 95)
        
        # Get container health
        results["container_health"] = await self.check_container_health()
        
        return results
    
    async def test_performance_calculation_api(self) -> Dict[str, Any]:
        """Test performance calculation API with varying portfolio sizes"""
        logger.info("Testing performance calculation API...")
        
        results = {
            "test_name": "performance_calculation_api",
            "test_timestamp": datetime.now().isoformat(),
            "portfolio_sizes": [1, 5, 10, 20],
            "test_results": [],
            "throughput_ops_per_sec": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            for portfolio_size in results["portfolio_sizes"]:
                logger.info(f"Testing portfolio size: {portfolio_size} symbols")
                
                # Prepare portfolio data
                symbols = list(self.test_data['portfolio'].keys())[:portfolio_size]
                portfolio_data = {
                    "positions": {},
                    "benchmark": "SPY",
                    "start_date": "2023-01-01",
                    "end_date": "2024-01-01"
                }
                
                for symbol in symbols:
                    portfolio_data["positions"][symbol] = {
                        "shares": np.random.randint(10, 1000),
                        "average_cost": np.random.uniform(50, 300)
                    }
                
                # Test multiple calculations
                response_times = []
                successful_requests = 0
                
                for i in range(5):  # 5 tests per portfolio size
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/analytics/performance/portfolio_{portfolio_size}_{i}",
                            json=portfolio_data,
                            timeout=30
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            response_times.append(response_time)
                            
                            if response.status == 200:
                                successful_requests += 1
                                data = await response.json()
                                logger.info(f"Portfolio {portfolio_size} calc {i+1}: {response_time:.2f}ms - {data.get('status')}")
                            else:
                                logger.warning(f"Portfolio {portfolio_size} calc {i+1}: HTTP {response.status}")
                    
                    except Exception as e:
                        logger.error(f"Portfolio {portfolio_size} calc {i+1} failed: {e}")
                    
                    await asyncio.sleep(0.1)
                
                # Calculate statistics for this portfolio size
                test_result = {
                    "portfolio_size": portfolio_size,
                    "successful_requests": successful_requests,
                    "total_requests": 5,
                    "success_rate": (successful_requests / 5) * 100,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0,
                    "p95_response_time": np.percentile(response_times, 95) if response_times else 0
                }
                
                results["test_results"].append(test_result)
        
        # Calculate overall throughput
        total_requests = sum(r["successful_requests"] for r in results["test_results"])
        total_time = sum(r["avg_response_time"] / 1000 for r in results["test_results"])
        if total_time > 0:
            results["throughput_ops_per_sec"] = total_requests / total_time
        
        return results
    
    async def test_risk_calculation_api(self) -> Dict[str, Any]:
        """Test risk analytics calculation API"""
        logger.info("Testing risk calculation API...")
        
        results = {
            "test_name": "risk_calculation_api",
            "test_timestamp": datetime.now().isoformat(),
            "test_results": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test risk calculations with different complexity levels
            risk_scenarios = [
                {"name": "simple_portfolio", "positions": 5, "var_confidence": 0.95},
                {"name": "medium_portfolio", "positions": 20, "var_confidence": 0.99},
                {"name": "complex_portfolio", "positions": 50, "var_confidence": 0.99},
            ]
            
            for scenario in risk_scenarios:
                logger.info(f"Testing risk scenario: {scenario['name']}")
                
                # Prepare risk calculation data
                risk_data = {
                    "portfolio_id": f"risk_test_{scenario['name']}",
                    "positions": {},
                    "var_confidence_level": scenario["var_confidence"],
                    "holding_period_days": 1,
                    "simulation_count": 10000
                }
                
                # Generate positions
                symbols = list(self.test_data['portfolio'].keys())[:scenario['positions']]
                for symbol in symbols:
                    risk_data["positions"][symbol] = {
                        "quantity": np.random.randint(-1000, 1000),  # Allow short positions
                        "market_value": np.random.uniform(10000, 100000)
                    }
                
                # Test multiple risk calculations
                response_times = []
                successful_requests = 0
                
                for i in range(3):  # 3 tests per scenario
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/analytics/risk/{scenario['name']}_{i}",
                            json=risk_data,
                            timeout=60  # Risk calculations can take longer
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            response_times.append(response_time)
                            
                            if response.status == 200:
                                successful_requests += 1
                                data = await response.json()
                                logger.info(f"Risk {scenario['name']} calc {i+1}: {response_time:.2f}ms - {data.get('status')}")
                            else:
                                logger.warning(f"Risk {scenario['name']} calc {i+1}: HTTP {response.status}")
                    
                    except Exception as e:
                        logger.error(f"Risk {scenario['name']} calc {i+1} failed: {e}")
                    
                    await asyncio.sleep(0.2)
                
                # Calculate statistics for this scenario
                test_result = {
                    "scenario_name": scenario['name'],
                    "positions_count": scenario['positions'],
                    "successful_requests": successful_requests,
                    "total_requests": 3,
                    "success_rate": (successful_requests / 3) * 100,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0
                }
                
                results["test_results"].append(test_result)
        
        return results
    
    async def test_concurrent_load(self, concurrent_users: int = 10, requests_per_user: int = 5) -> Dict[str, Any]:
        """Test concurrent load handling"""
        logger.info(f"Testing concurrent load: {concurrent_users} users, {requests_per_user} requests each")
        
        results = {
            "test_name": "concurrent_load",
            "test_timestamp": datetime.now().isoformat(),
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": concurrent_users * requests_per_user,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": []
        }
        
        async def user_session(user_id: int):
            """Simulate a single user making multiple requests"""
            user_results = {"successful": 0, "failed": 0, "response_times": [], "errors": []}
            
            async with aiohttp.ClientSession() as session:
                for req_id in range(requests_per_user):
                    # Mix different types of requests
                    if req_id % 3 == 0:
                        # Health check
                        endpoint = "/health"
                        payload = None
                    elif req_id % 3 == 1:
                        # Performance calculation
                        endpoint = f"/analytics/performance/concurrent_test_user{user_id}_req{req_id}"
                        payload = {
                            "positions": {"AAPL": {"shares": 100, "average_cost": 150.0}},
                            "benchmark": "SPY"
                        }
                    else:
                        # Risk calculation
                        endpoint = f"/analytics/risk/concurrent_test_user{user_id}_req{req_id}"
                        payload = {
                            "positions": {"AAPL": {"quantity": 100, "market_value": 15000}},
                            "var_confidence_level": 0.95
                        }
                    
                    start_time = time.time()
                    try:
                        if payload:
                            async with session.post(f"{self.base_url}{endpoint}", json=payload, timeout=30) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                    user_results["errors"].append(f"HTTP {response.status}")
                        else:
                            async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                    user_results["errors"].append(f"HTTP {response.status}")
                    
                    except Exception as e:
                        user_results["failed"] += 1
                        user_results["errors"].append(str(e))
                    
                    await asyncio.sleep(0.1)  # Small delay between requests
            
            return user_results
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [user_session(user_id) for user_id in range(concurrent_users)]
        user_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        total_test_time = time.time() - start_time
        
        # Aggregate results
        all_response_times = []
        for user_result in user_results_list:
            if isinstance(user_result, dict):
                results["successful_requests"] += user_result["successful"]
                results["failed_requests"] += user_result["failed"]
                all_response_times.extend(user_result["response_times"])
                results["errors"].extend(user_result["errors"])
        
        results["response_times"] = all_response_times
        
        # Calculate statistics
        if all_response_times:
            results["avg_response_time"] = statistics.mean(all_response_times)
            results["min_response_time"] = min(all_response_times)
            results["max_response_time"] = max(all_response_times)
            results["p50_response_time"] = np.percentile(all_response_times, 50)
            results["p95_response_time"] = np.percentile(all_response_times, 95)
            results["p99_response_time"] = np.percentile(all_response_times, 99)
        
        results["success_rate"] = (results["successful_requests"] / results["total_requests"]) * 100
        results["requests_per_second"] = results["total_requests"] / total_test_time
        results["total_test_time_seconds"] = total_test_time
        
        logger.info(f"Concurrent load test completed: {results['success_rate']:.1f}% success rate, "
                   f"{results['requests_per_second']:.1f} requests/sec")
        
        return results
    
    async def test_memory_and_cpu_usage(self) -> Dict[str, Any]:
        """Monitor memory and CPU usage during operations"""
        logger.info("Testing memory and CPU usage patterns...")
        
        results = {
            "test_name": "memory_cpu_usage",
            "test_timestamp": datetime.now().isoformat(),
            "baseline_metrics": {},
            "load_metrics": {},
            "peak_metrics": {},
            "usage_timeline": []
        }
        
        # Get baseline metrics
        baseline_container_stats = await self.check_container_health()
        results["baseline_metrics"] = baseline_container_stats
        
        # Monitor during load test
        monitoring_task = asyncio.create_task(self._monitor_resources(results["usage_timeline"]))
        
        # Run a load test while monitoring
        load_results = await self.test_concurrent_load(concurrent_users=5, requests_per_user=10)
        
        # Stop monitoring
        monitoring_task.cancel()
        
        # Get peak metrics
        if results["usage_timeline"]:
            cpu_values = [m["cpu_percent"] for m in results["usage_timeline"]]
            memory_values = [m["memory_usage_mb"] for m in results["usage_timeline"]]
            
            results["peak_metrics"] = {
                "peak_cpu_percent": max(cpu_values),
                "peak_memory_mb": max(memory_values),
                "avg_cpu_percent": statistics.mean(cpu_values),
                "avg_memory_mb": statistics.mean(memory_values)
            }
        
        results["load_test_results"] = load_results
        
        return results
    
    async def _monitor_resources(self, timeline: List[Dict[str, Any]], interval: float = 1.0):
        """Monitor container resources at regular intervals"""
        try:
            while True:
                stats = await self.check_container_health()
                stats["timestamp"] = datetime.now().isoformat()
                timeline.append(stats)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all analytics engine performance tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE ANALYTICS ENGINE PERFORMANCE TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "Analytics Engine Comprehensive Performance Test",
            "test_start_time": datetime.now().isoformat(),
            "engine_url": self.base_url,
            "container_name": self.container_name,
            "test_results": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Test sequence
        test_sequence = [
            ("health_endpoint", self.test_health_endpoint),
            ("performance_calculation_api", self.test_performance_calculation_api),
            ("risk_calculation_api", self.test_risk_calculation_api),
            ("concurrent_load_test", lambda: self.test_concurrent_load(concurrent_users=8, requests_per_user=5)),
            ("memory_cpu_usage", self.test_memory_and_cpu_usage)
        ]
        
        for test_name, test_function in test_sequence:
            logger.info(f"Running test: {test_name}")
            try:
                test_result = await test_function()
                comprehensive_results["test_results"][test_name] = test_result
                logger.info(f"Completed test: {test_name} ✅")
            except Exception as e:
                logger.error(f"Failed test: {test_name} ❌ - {e}")
                comprehensive_results["test_results"][test_name] = {
                    "error": str(e),
                    "test_failed": True
                }
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Calculate performance summary
        total_test_time = time.time() - start_time
        comprehensive_results["total_test_time_seconds"] = total_test_time
        comprehensive_results["test_end_time"] = datetime.now().isoformat()
        
        # Analyze results and create performance summary
        self._create_performance_summary(comprehensive_results)
        
        # Generate recommendations
        self._generate_recommendations(comprehensive_results)
        
        logger.info("="*80)
        logger.info("ANALYTICS ENGINE PERFORMANCE TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_performance_summary(self, results: Dict[str, Any]):
        """Create a performance summary from all test results"""
        summary = {
            "overall_health": "unknown",
            "api_response_performance": {},
            "scalability_metrics": {},
            "resource_utilization": {}
        }
        
        # Health endpoint analysis
        health_test = results["test_results"].get("health_endpoint", {})
        if not health_test.get("test_failed"):
            summary["api_response_performance"]["health_avg_ms"] = health_test.get("avg_response_time", 0)
            summary["api_response_performance"]["health_success_rate"] = health_test.get("success_rate", 0)
        
        # Performance calculation analysis
        perf_test = results["test_results"].get("performance_calculation_api", {})
        if not perf_test.get("test_failed"):
            summary["api_response_performance"]["performance_calc_throughput"] = perf_test.get("throughput_ops_per_sec", 0)
        
        # Concurrent load analysis
        load_test = results["test_results"].get("concurrent_load_test", {})
        if not load_test.get("test_failed"):
            summary["scalability_metrics"]["concurrent_users_supported"] = load_test.get("concurrent_users", 0)
            summary["scalability_metrics"]["requests_per_second"] = load_test.get("requests_per_second", 0)
            summary["scalability_metrics"]["concurrent_success_rate"] = load_test.get("success_rate", 0)
        
        # Resource utilization analysis
        resource_test = results["test_results"].get("memory_cpu_usage", {})
        if not resource_test.get("test_failed"):
            peak_metrics = resource_test.get("peak_metrics", {})
            summary["resource_utilization"]["peak_cpu_percent"] = peak_metrics.get("peak_cpu_percent", 0)
            summary["resource_utilization"]["peak_memory_mb"] = peak_metrics.get("peak_memory_mb", 0)
            summary["resource_utilization"]["avg_cpu_percent"] = peak_metrics.get("avg_cpu_percent", 0)
            summary["resource_utilization"]["avg_memory_mb"] = peak_metrics.get("avg_memory_mb", 0)
        
        # Overall health assessment
        success_rates = []
        for test_name, test_result in results["test_results"].items():
            if not test_result.get("test_failed"):
                if "success_rate" in test_result:
                    success_rates.append(test_result["success_rate"])
        
        if success_rates:
            overall_success = statistics.mean(success_rates)
            if overall_success >= 95:
                summary["overall_health"] = "excellent"
            elif overall_success >= 90:
                summary["overall_health"] = "good"
            elif overall_success >= 80:
                summary["overall_health"] = "fair"
            else:
                summary["overall_health"] = "poor"
        
        results["performance_summary"] = summary
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate performance recommendations based on test results"""
        recommendations = []
        
        summary = results.get("performance_summary", {})
        
        # API response time recommendations
        api_perf = summary.get("api_response_performance", {})
        health_avg = api_perf.get("health_avg_ms", 0)
        if health_avg > 100:
            recommendations.append({
                "category": "response_time",
                "severity": "medium",
                "issue": f"Health endpoint average response time is {health_avg:.1f}ms",
                "recommendation": "Consider optimizing health check logic or increasing container resources"
            })
        
        # Scalability recommendations
        scalability = summary.get("scalability_metrics", {})
        success_rate = scalability.get("concurrent_success_rate", 100)
        if success_rate < 95:
            recommendations.append({
                "category": "scalability",
                "severity": "high",
                "issue": f"Concurrent load success rate is only {success_rate:.1f}%",
                "recommendation": "Increase container resources or implement connection pooling"
            })
        
        rps = scalability.get("requests_per_second", 0)
        if rps < 10:
            recommendations.append({
                "category": "throughput",
                "severity": "medium",
                "issue": f"Low throughput: {rps:.1f} requests/second",
                "recommendation": "Consider scaling horizontally or optimizing request handling"
            })
        
        # Resource utilization recommendations
        resources = summary.get("resource_utilization", {})
        peak_cpu = resources.get("peak_cpu_percent", 0)
        if peak_cpu > 80:
            recommendations.append({
                "category": "cpu_usage",
                "severity": "high",
                "issue": f"High CPU usage: {peak_cpu:.1f}%",
                "recommendation": "Increase CPU limits or optimize CPU-intensive operations"
            })
        
        peak_memory = resources.get("peak_memory_mb", 0)
        if peak_memory > 1024:  # > 1GB
            recommendations.append({
                "category": "memory_usage",
                "severity": "medium",
                "issue": f"High memory usage: {peak_memory:.1f}MB",
                "recommendation": "Monitor for memory leaks or increase memory limits"
            })
        
        # Overall health recommendations
        overall_health = summary.get("overall_health", "unknown")
        if overall_health == "poor":
            recommendations.append({
                "category": "general",
                "severity": "high",
                "issue": "Overall system health is poor",
                "recommendation": "Review all failed tests and address critical issues immediately"
            })
        elif overall_health == "fair":
            recommendations.append({
                "category": "general",
                "severity": "medium",
                "issue": "System health needs improvement",
                "recommendation": "Address performance bottlenecks and optimize resource usage"
            })
        
        if not recommendations:
            recommendations.append({
                "category": "general",
                "severity": "info",
                "issue": "No critical issues detected",
                "recommendation": "Analytics Engine is performing well. Continue monitoring for optimal performance."
            })
        
        results["recommendations"] = recommendations

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/analytics_engine_performance_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = AnalyticsEnginePerformanceTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYTICS ENGINE PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
    
    api_perf = summary.get("api_response_performance", {})
    if api_perf:
        print(f"\nAPI Performance:")
        print(f"  Health Endpoint Avg Response: {api_perf.get('health_avg_ms', 0):.1f}ms")
        print(f"  Performance Calc Throughput: {api_perf.get('performance_calc_throughput', 0):.1f} ops/sec")
    
    scalability = summary.get("scalability_metrics", {})
    if scalability:
        print(f"\nScalability Metrics:")
        print(f"  Concurrent Users Tested: {scalability.get('concurrent_users_supported', 0)}")
        print(f"  Requests per Second: {scalability.get('requests_per_second', 0):.1f}")
        print(f"  Success Rate: {scalability.get('concurrent_success_rate', 0):.1f}%")
    
    resources = summary.get("resource_utilization", {})
    if resources:
        print(f"\nResource Utilization:")
        print(f"  Peak CPU: {resources.get('peak_cpu_percent', 0):.1f}%")
        print(f"  Peak Memory: {resources.get('peak_memory_mb', 0):.1f}MB")
        print(f"  Average CPU: {resources.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Average Memory: {resources.get('avg_memory_mb', 0):.1f}MB")
    
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            severity = rec.get("severity", "info").upper()
            print(f"  {i}. [{severity}] {rec.get('recommendation', 'N/A')}")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
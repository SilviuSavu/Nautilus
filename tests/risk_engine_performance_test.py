#!/usr/bin/env python3
"""
Risk Engine Comprehensive Performance Test
Tests the containerized Risk Engine performance, capabilities, and integration
Focuses on risk management, ML-based breach detection, and professional analytics
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

class RiskEnginePerformanceTest:
    """
    Comprehensive performance testing suite for Risk Engine
    Tests containerized risk management, ML-based analytics, and professional reporting
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8200"
        self.docker_client = docker.from_env()
        self.container_name = "nautilus-risk-engine"
        self.test_data = {}
        self.performance_metrics = {}
        self.load_test_data()
        
    def load_test_data(self):
        """Load comprehensive test data for risk engine testing"""
        logger.info("Loading test data for Risk Engine performance testing...")
        
        try:
            # Load portfolio data for risk calculations
            portfolio_data = {}
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'UNH']
            
            for symbol in symbols:
                try:
                    with open(f'/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_{symbol}_daily.json', 'r') as f:
                        data = json.load(f)
                        portfolio_data[symbol] = data
                except FileNotFoundError:
                    logger.warning(f"Could not load data for {symbol}")
            
            self.test_data['portfolio'] = portfolio_data
            
            # Generate returns data for risk analytics
            self.generate_returns_data()
            
            # Load high-frequency data for real-time testing
            try:
                with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/synthetic_AAPL_hf.json', 'r') as f:
                    hf_data = json.load(f)
                    self.test_data['high_frequency'] = hf_data[:1000]  # First 1000 records
            except FileNotFoundError:
                self.generate_synthetic_hf_data()
            
            logger.info(f"Loaded test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.generate_synthetic_test_data()
    
    def generate_returns_data(self):
        """Generate realistic returns data from price data"""
        returns_data = {}
        
        for symbol, price_data in self.test_data['portfolio'].items():
            if not price_data:
                continue
                
            # Extract closing prices and calculate returns
            dates = sorted(price_data.keys())
            prices = [price_data[date]['Close'] for date in dates]
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            returns_data[symbol] = returns
        
        self.test_data['returns'] = returns_data
        
        # Generate portfolio returns (equal weighted for testing)
        if returns_data:
            min_length = min(len(returns) for returns in returns_data.values())
            portfolio_returns = []
            
            for i in range(min_length):
                daily_returns = [returns_data[symbol][i] for symbol in returns_data.keys()]
                portfolio_return = np.mean(daily_returns)
                portfolio_returns.append(portfolio_return)
            
            self.test_data['portfolio_returns'] = portfolio_returns
            
            # Generate benchmark returns (simulate SPY)
            benchmark_returns = [ret * 0.8 + np.random.normal(0, 0.001) for ret in portfolio_returns]
            self.test_data['benchmark_returns'] = benchmark_returns
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data for risk testing...")
        
        np.random.seed(42)
        
        # Generate portfolio data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        portfolio_data = {}
        returns_data = {}
        
        for symbol in symbols:
            # Generate 2 years of daily returns
            n_days = 504  # ~2 years
            returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% mean, 2% vol
            returns_data[symbol] = returns.tolist()
            
            # Generate corresponding price data
            base_price = np.random.uniform(50, 300)
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            # Create price data structure
            dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
            price_dict = {}
            
            for i, date in enumerate(dates):
                price_dict[date.isoformat()] = {
                    'Open': prices[i] * (1 + np.random.normal(0, 0.001)),
                    'High': prices[i] * (1 + abs(np.random.normal(0, 0.005))),
                    'Low': prices[i] * (1 - abs(np.random.normal(0, 0.005))),
                    'Close': prices[i],
                    'Volume': int(np.random.exponential(1000000)),
                    'Adj Close': prices[i]
                }
            
            portfolio_data[symbol] = price_dict
        
        self.test_data['portfolio'] = portfolio_data
        self.test_data['returns'] = returns_data
        
        # Generate portfolio and benchmark returns
        portfolio_returns = np.mean(list(returns_data.values()), axis=0).tolist()
        benchmark_returns = [ret * 0.8 + np.random.normal(0, 0.001) for ret in portfolio_returns]
        
        self.test_data['portfolio_returns'] = portfolio_returns
        self.test_data['benchmark_returns'] = benchmark_returns
        
        self.generate_synthetic_hf_data()
    
    def generate_synthetic_hf_data(self):
        """Generate synthetic high-frequency data for real-time testing"""
        np.random.seed(42)
        hf_data = []
        
        base_time = datetime.now() - timedelta(hours=1)
        for i in range(1000):
            timestamp = base_time + timedelta(seconds=i * 3.6)  # 1-second intervals
            price = 150.0 + np.random.normal(0, 0.5)
            
            hf_data.append({
                'timestamp': timestamp.isoformat(),
                'symbol': 'AAPL',
                'price': round(price, 2),
                'volume': int(np.random.exponential(10000))
            })
        
        self.test_data['high_frequency'] = hf_data
    
    async def check_container_health(self) -> Dict[str, Any]:
        """Check if the Risk Engine container is healthy"""
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
        """Test the health endpoint and ML model status"""
        logger.info("Testing Risk Engine health endpoint...")
        
        results = {
            "test_name": "health_endpoint",
            "test_timestamp": datetime.now().isoformat(),
            "response_times": [],
            "success_rate": 0.0,
            "ml_model_status": "unknown",
            "pyfolio_status": "unknown",
            "container_health": {}
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}/health", timeout=10) as response:
                        response_time = (time.time() - start_time) * 1000  # Convert to ms
                        results["response_times"].append(response_time)
                        
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Health check {i+1}: {response_time:.2f}ms - {data.get('status', 'unknown')}")
                            
                            # Extract additional status info
                            if i == 0:  # First successful response
                                results["ml_model_status"] = data.get("ml_model_status", "unknown")
                                pyfolio_integration = data.get("pyfolio_integration", {})
                                results["pyfolio_status"] = pyfolio_integration.get("status", "unknown")
                                results["active_limits"] = data.get("active_limits", 0)
                                results["active_breaches"] = data.get("active_breaches", 0)
                        else:
                            logger.warning(f"Health check {i+1}: HTTP {response.status}")
                
                except Exception as e:
                    logger.error(f"Health check {i+1} failed: {e}")
                
                await asyncio.sleep(0.1)
        
        # Calculate statistics
        if results["response_times"]:
            results["success_rate"] = len(results["response_times"]) / 10 * 100
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
        
        results["container_health"] = await self.check_container_health()
        
        return results
    
    async def test_risk_limits_management(self) -> Dict[str, Any]:
        """Test risk limits creation and management"""
        logger.info("Testing risk limits management...")
        
        results = {
            "test_name": "risk_limits_management",
            "test_timestamp": datetime.now().isoformat(),
            "limits_created": 0,
            "creation_times": [],
            "check_times": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Create various types of risk limits
            limit_types = [
                {"type": "position_size", "value": 100000},
                {"type": "portfolio_value", "value": 1000000},
                {"type": "daily_loss", "value": 50000},
                {"type": "concentration", "value": 0.3},
                {"type": "var_limit", "value": 100000}
            ]
            
            for i, limit_config in enumerate(limit_types):
                limit_data = {
                    "limit_id": f"test_limit_{limit_config['type']}_{int(time.time())}",
                    "limit_type": limit_config["type"],
                    "limit_value": limit_config["value"],
                    "current_value": limit_config["value"] * 0.5,  # 50% of limit
                    "threshold_warning": 0.8,
                    "threshold_breach": 1.0,
                    "enabled": True,
                    "portfolio_id": f"test_portfolio_{i}"
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/risk/limits",
                        json=limit_data,
                        timeout=10
                    ) as response:
                        creation_time = (time.time() - start_time) * 1000
                        results["creation_times"].append(creation_time)
                        
                        if response.status == 200:
                            results["limits_created"] += 1
                            data = await response.json()
                            logger.info(f"Created limit {limit_config['type']}: {creation_time:.2f}ms")
                        else:
                            logger.warning(f"Failed to create limit {limit_config['type']}: HTTP {response.status}")
                
                except Exception as e:
                    logger.error(f"Error creating limit {limit_config['type']}: {e}")
                
                await asyncio.sleep(0.1)
        
        # Calculate statistics
        if results["creation_times"]:
            results["avg_creation_time"] = statistics.mean(results["creation_times"])
            results["max_creation_time"] = max(results["creation_times"])
        
        return results
    
    async def test_risk_check_performance(self) -> Dict[str, Any]:
        """Test risk check performance with varying portfolio sizes"""
        logger.info("Testing risk check performance...")
        
        results = {
            "test_name": "risk_check_performance",
            "test_timestamp": datetime.now().isoformat(),
            "portfolio_sizes": [1, 5, 10, 20, 50],
            "test_results": [],
            "throughput_checks_per_sec": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            for portfolio_size in results["portfolio_sizes"]:
                logger.info(f"Testing portfolio size: {portfolio_size} positions")
                
                # Create position data
                symbols = list(self.test_data['portfolio'].keys())[:portfolio_size]
                position_data = {
                    "positions": {},
                    "total_value": 0,
                    "timestamp": time.time_ns()
                }
                
                total_value = 0
                for symbol in symbols:
                    shares = np.random.randint(100, 10000)
                    price = np.random.uniform(50, 300)
                    position_value = shares * price
                    total_value += position_value
                    
                    position_data["positions"][symbol] = {
                        "shares": shares,
                        "price": price,
                        "market_value": position_value,
                        "unrealized_pnl": np.random.normal(0, position_value * 0.1)
                    }
                
                position_data["total_value"] = total_value
                
                # Test multiple risk checks
                response_times = []
                successful_checks = 0
                
                for i in range(5):  # 5 checks per portfolio size
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/risk/check/portfolio_{portfolio_size}_{i}",
                            json=position_data,
                            timeout=30
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            response_times.append(response_time)
                            
                            if response.status == 200:
                                successful_checks += 1
                                data = await response.json()
                                logger.info(f"Risk check {portfolio_size} pos #{i+1}: {response_time:.2f}ms - {data.get('status')}")
                            else:
                                logger.warning(f"Risk check {portfolio_size} pos #{i+1}: HTTP {response.status}")
                    
                    except Exception as e:
                        logger.error(f"Risk check {portfolio_size} pos #{i+1} failed: {e}")
                    
                    await asyncio.sleep(0.1)
                
                # Calculate statistics for this portfolio size
                test_result = {
                    "portfolio_size": portfolio_size,
                    "successful_checks": successful_checks,
                    "total_checks": 5,
                    "success_rate": (successful_checks / 5) * 100,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0
                }
                
                results["test_results"].append(test_result)
        
        # Calculate overall throughput
        total_successful = sum(r["successful_checks"] for r in results["test_results"])
        total_time = sum(r["avg_response_time"] / 1000 for r in results["test_results"])
        if total_time > 0:
            results["throughput_checks_per_sec"] = total_successful / total_time
        
        return results
    
    async def test_pyfolio_analytics(self) -> Dict[str, Any]:
        """Test PyFolio analytics integration"""
        logger.info("Testing PyFolio analytics integration...")
        
        results = {
            "test_name": "pyfolio_analytics",
            "test_timestamp": datetime.now().isoformat(),
            "analytics_tests": [],
            "tear_sheet_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test PyFolio analytics computation
            portfolio_returns = self.test_data.get('portfolio_returns', [])[:252]  # 1 year
            benchmark_returns = self.test_data.get('benchmark_returns', [])[:252]
            
            if not portfolio_returns:
                logger.warning("No portfolio returns data available for PyFolio testing")
                return results
            
            analytics_data = {
                "returns": portfolio_returns,
                "benchmark_returns": benchmark_returns,
                "risk_free_rate": 0.02
            }
            
            # Test analytics computation
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/risk/analytics/pyfolio/test_portfolio",
                    json=analytics_data,
                    timeout=60
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        analytics_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "metrics_computed": len(data.get("analytics", {})),
                            "sharpe_ratio": data.get("analytics", {}).get("sharpe_ratio"),
                            "max_drawdown": data.get("analytics", {}).get("max_drawdown"),
                            "volatility": data.get("analytics", {}).get("volatility")
                        }
                        logger.info(f"PyFolio analytics: {response_time:.2f}ms, {analytics_result['metrics_computed']} metrics")
                    else:
                        analytics_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                analytics_result = {"success": False, "error": str(e)}
            
            results["analytics_tests"].append(analytics_result)
            
            # Test tear sheet generation
            tear_sheet_data = {
                "returns": portfolio_returns[:100],  # Smaller dataset for tear sheet
                "benchmark_returns": benchmark_returns[:100],
                "format": "json"
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.base_url}/risk/analytics/pyfolio/tear-sheet/test_portfolio",
                    json=tear_sheet_data,
                    timeout=60
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        tear_sheet_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "sections_generated": len(data.get("tear_sheet", {})),
                            "format": data.get("format", "unknown")
                        }
                        logger.info(f"PyFolio tear sheet: {response_time:.2f}ms, {tear_sheet_result['sections_generated']} sections")
                    else:
                        tear_sheet_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                tear_sheet_result = {"success": False, "error": str(e)}
            
            results["tear_sheet_tests"].append(tear_sheet_result)
        
        return results
    
    async def test_professional_reporting(self) -> Dict[str, Any]:
        """Test professional risk reporting capabilities"""
        logger.info("Testing professional risk reporting...")
        
        results = {
            "test_name": "professional_reporting",
            "test_timestamp": datetime.now().isoformat(),
            "report_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test different report types
            report_types = ["executive_summary", "comprehensive", "risk_focused"]
            
            for report_type in report_types:
                logger.info(f"Testing {report_type} report generation...")
                
                start_time = time.time()
                try:
                    # Use query parameters for GET request
                    url = f"{self.base_url}/risk/analytics/professional/test_portfolio"
                    params = {
                        "report_type": report_type,
                        "format": "json",
                        "date_range_days": 252,
                        "benchmark_symbol": "SPY"
                    }
                    
                    async with session.post(url, params=params, timeout=120) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            report_result = {
                                "report_type": report_type,
                                "success": True,
                                "response_time_ms": response_time,
                                "report_sections": len(data.get("report", {})),
                                "format": data.get("format", "unknown")
                            }
                            logger.info(f"Professional report {report_type}: {response_time:.2f}ms")
                        else:
                            report_result = {
                                "report_type": report_type,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    report_result = {
                        "report_type": report_type,
                        "success": False,
                        "error": str(e)
                    }
                
                results["report_tests"].append(report_result)
                await asyncio.sleep(1)  # Brief delay between report generations
        
        return results
    
    async def test_concurrent_risk_operations(self, concurrent_users: int = 5, operations_per_user: int = 3) -> Dict[str, Any]:
        """Test concurrent risk operations"""
        logger.info(f"Testing concurrent risk operations: {concurrent_users} users, {operations_per_user} ops each")
        
        results = {
            "test_name": "concurrent_risk_operations",
            "test_timestamp": datetime.now().isoformat(),
            "concurrent_users": concurrent_users,
            "operations_per_user": operations_per_user,
            "total_operations": concurrent_users * operations_per_user,
            "successful_operations": 0,
            "failed_operations": 0,
            "response_times": [],
            "operation_breakdown": {}
        }
        
        async def user_operations(user_id: int):
            """Simulate a user performing multiple risk operations"""
            user_results = {"successful": 0, "failed": 0, "response_times": [], "operations": []}
            
            async with aiohttp.ClientSession() as session:
                for op_id in range(operations_per_user):
                    # Rotate between different operation types
                    if op_id % 3 == 0:
                        # Health check
                        operation_type = "health_check"
                        start_time = time.time()
                        try:
                            async with session.get(f"{self.base_url}/health", timeout=10) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["operations"].append({"type": operation_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["operations"].append({"type": operation_type, "success": False})
                    
                    elif op_id % 3 == 1:
                        # Risk check
                        operation_type = "risk_check"
                        position_data = {
                            "positions": {"AAPL": {"shares": 100, "price": 150, "market_value": 15000}},
                            "total_value": 15000
                        }
                        start_time = time.time()
                        try:
                            async with session.post(
                                f"{self.base_url}/risk/check/concurrent_user{user_id}_op{op_id}",
                                json=position_data,
                                timeout=30
                            ) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["operations"].append({"type": operation_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["operations"].append({"type": operation_type, "success": False})
                    
                    else:
                        # PyFolio analytics
                        operation_type = "pyfolio_analytics"
                        analytics_data = {
                            "returns": self.test_data.get('portfolio_returns', [])[:50],  # Small dataset
                            "risk_free_rate": 0.02
                        }
                        start_time = time.time()
                        try:
                            async with session.post(
                                f"{self.base_url}/risk/analytics/pyfolio/concurrent_user{user_id}_op{op_id}",
                                json=analytics_data,
                                timeout=60
                            ) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["operations"].append({"type": operation_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["operations"].append({"type": operation_type, "success": False})
                    
                    await asyncio.sleep(0.2)  # Brief delay between operations
            
            return user_results
        
        # Run concurrent operations
        start_time = time.time()
        tasks = [user_operations(user_id) for user_id in range(concurrent_users)]
        user_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        total_test_time = time.time() - start_time
        
        # Aggregate results
        operation_counts = {}
        for user_result in user_results_list:
            if isinstance(user_result, dict):
                results["successful_operations"] += user_result["successful"]
                results["failed_operations"] += user_result["failed"]
                results["response_times"].extend(user_result["response_times"])
                
                # Count operation types
                for op in user_result["operations"]:
                    op_type = op["type"]
                    if op_type not in operation_counts:
                        operation_counts[op_type] = {"successful": 0, "failed": 0}
                    
                    if op["success"]:
                        operation_counts[op_type]["successful"] += 1
                    else:
                        operation_counts[op_type]["failed"] += 1
        
        results["operation_breakdown"] = operation_counts
        
        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["p95_response_time"] = np.percentile(results["response_times"], 95)
        
        results["success_rate"] = (results["successful_operations"] / results["total_operations"]) * 100
        results["operations_per_second"] = results["total_operations"] / total_test_time
        results["total_test_time_seconds"] = total_test_time
        
        logger.info(f"Concurrent operations test completed: {results['success_rate']:.1f}% success rate, "
                   f"{results['operations_per_second']:.1f} ops/sec")
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all risk engine performance tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE RISK ENGINE PERFORMANCE TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "Risk Engine Comprehensive Performance Test",
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
            ("risk_limits_management", self.test_risk_limits_management),
            ("risk_check_performance", self.test_risk_check_performance),
            ("pyfolio_analytics", self.test_pyfolio_analytics),
            ("professional_reporting", self.test_professional_reporting),
            ("concurrent_risk_operations", lambda: self.test_concurrent_risk_operations(concurrent_users=3, operations_per_user=4))
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
        
        logger.info("="*80)
        logger.info("RISK ENGINE PERFORMANCE TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_performance_summary(self, results: Dict[str, Any]):
        """Create a performance summary from all test results"""
        summary = {
            "overall_health": "unknown",
            "risk_management_performance": {},
            "analytics_performance": {},
            "ml_integration_status": "unknown",
            "resource_efficiency": {}
        }
        
        # Health and ML status analysis
        health_test = results["test_results"].get("health_endpoint", {})
        if not health_test.get("test_failed"):
            summary["ml_integration_status"] = health_test.get("ml_model_status", "unknown")
            summary["pyfolio_integration"] = health_test.get("pyfolio_status", "unknown")
            summary["active_limits"] = health_test.get("active_limits", 0)
            summary["active_breaches"] = health_test.get("active_breaches", 0)
        
        # Risk check performance analysis
        risk_check_test = results["test_results"].get("risk_check_performance", {})
        if not risk_check_test.get("test_failed"):
            summary["risk_management_performance"]["throughput_checks_per_sec"] = risk_check_test.get("throughput_checks_per_sec", 0)
            
            # Average response time across portfolio sizes
            test_results = risk_check_test.get("test_results", [])
            if test_results:
                avg_response_times = [r["avg_response_time"] for r in test_results if r["avg_response_time"] > 0]
                if avg_response_times:
                    summary["risk_management_performance"]["avg_check_time_ms"] = statistics.mean(avg_response_times)
        
        # Analytics performance analysis
        pyfolio_test = results["test_results"].get("pyfolio_analytics", {})
        if not pyfolio_test.get("test_failed"):
            analytics_tests = pyfolio_test.get("analytics_tests", [])
            if analytics_tests and analytics_tests[0].get("success"):
                summary["analytics_performance"]["pyfolio_response_time_ms"] = analytics_tests[0].get("response_time_ms", 0)
                summary["analytics_performance"]["metrics_computed"] = analytics_tests[0].get("metrics_computed", 0)
        
        # Professional reporting performance
        reporting_test = results["test_results"].get("professional_reporting", {})
        if not reporting_test.get("test_failed"):
            report_tests = reporting_test.get("report_tests", [])
            successful_reports = [r for r in report_tests if r.get("success")]
            if successful_reports:
                avg_report_time = statistics.mean([r["response_time_ms"] for r in successful_reports])
                summary["analytics_performance"]["professional_report_avg_time_ms"] = avg_report_time
        
        # Concurrent operations analysis
        concurrent_test = results["test_results"].get("concurrent_risk_operations", {})
        if not concurrent_test.get("test_failed"):
            summary["resource_efficiency"]["concurrent_success_rate"] = concurrent_test.get("success_rate", 0)
            summary["resource_efficiency"]["operations_per_second"] = concurrent_test.get("operations_per_second", 0)
        
        # Overall health assessment
        success_indicators = []
        if summary["ml_integration_status"] == "loaded":
            success_indicators.append(1)
        if summary["risk_management_performance"].get("throughput_checks_per_sec", 0) > 1:
            success_indicators.append(1)
        if summary["resource_efficiency"].get("concurrent_success_rate", 0) >= 90:
            success_indicators.append(1)
        
        if len(success_indicators) >= 2:
            summary["overall_health"] = "good"
        elif len(success_indicators) >= 1:
            summary["overall_health"] = "fair"
        else:
            summary["overall_health"] = "poor"
        
        results["performance_summary"] = summary

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/risk_engine_performance_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = RiskEnginePerformanceTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("RISK ENGINE PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
    print(f"ML Integration: {summary.get('ml_integration_status', 'unknown')}")
    print(f"PyFolio Status: {summary.get('pyfolio_integration', 'unknown')}")
    print(f"Active Limits: {summary.get('active_limits', 0)}")
    print(f"Active Breaches: {summary.get('active_breaches', 0)}")
    
    risk_perf = summary.get("risk_management_performance", {})
    if risk_perf:
        print(f"\nRisk Management Performance:")
        print(f"  Risk Checks Throughput: {risk_perf.get('throughput_checks_per_sec', 0):.1f} checks/sec")
        print(f"  Average Check Time: {risk_perf.get('avg_check_time_ms', 0):.1f}ms")
    
    analytics_perf = summary.get("analytics_performance", {})
    if analytics_perf:
        print(f"\nAnalytics Performance:")
        print(f"  PyFolio Response Time: {analytics_perf.get('pyfolio_response_time_ms', 0):.1f}ms")
        print(f"  Professional Reports Avg: {analytics_perf.get('professional_report_avg_time_ms', 0):.1f}ms")
    
    resource_eff = summary.get("resource_efficiency", {})
    if resource_eff:
        print(f"\nResource Efficiency:")
        print(f"  Concurrent Success Rate: {resource_eff.get('concurrent_success_rate', 0):.1f}%")
        print(f"  Operations per Second: {resource_eff.get('operations_per_second', 0):.1f}")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
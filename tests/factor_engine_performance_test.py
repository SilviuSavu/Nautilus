#!/usr/bin/env python3
"""
Factor Engine Comprehensive Performance Test
Tests the containerized Factor Engine performance, capabilities, and integration
Focuses on factor synthesis, 485 factor definitions, and parallel processing
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

class FactorEnginePerformanceTest:
    """
    Comprehensive performance testing suite for Factor Engine
    Tests containerized factor synthesis, 485+ factor definitions, and high-volume processing
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8300"
        self.docker_client = docker.from_env()
        self.container_name = "nautilus-factor-engine"
        self.test_data = {}
        self.performance_metrics = {}
        self.load_test_data()
        
    def load_test_data(self):
        """Load comprehensive test data for factor engine testing"""
        logger.info("Loading test data for Factor Engine performance testing...")
        
        try:
            # Load portfolio data for factor calculations
            portfolio_data = {}
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'UNH',
                      'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'NFLX', 'CRM', 'CMCSA', 'XOM', 'NVDA']
            
            for symbol in symbols:
                try:
                    with open(f'/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_{symbol}_daily.json', 'r') as f:
                        data = json.load(f)
                        portfolio_data[symbol] = data
                except FileNotFoundError:
                    logger.warning(f"Could not load data for {symbol}")
            
            self.test_data['portfolio'] = portfolio_data
            
            # Generate synthetic factor data for testing
            self.generate_factor_test_data()
            
            logger.info(f"Loaded test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.generate_synthetic_test_data()
    
    def generate_factor_test_data(self):
        """Generate synthetic factor data for comprehensive testing"""
        np.random.seed(42)
        
        # Generate various factor categories for testing
        factor_categories = {
            'technical': ['RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'SMA_20', 'EMA_12', 'STOCH_K', 'STOCH_D', 'ATR', 'ADX'],
            'fundamental': ['PE_RATIO', 'PB_RATIO', 'ROE', 'ROA', 'DEBT_TO_EQUITY', 'CURRENT_RATIO', 'QUICK_RATIO', 'EPS_GROWTH'],
            'macroeconomic': ['GDP_GROWTH', 'INFLATION_RATE', 'UNEMPLOYMENT', 'FED_RATE', 'YIELD_CURVE', 'DXY_INDEX'],
            'sentiment': ['VIX_LEVEL', 'PUT_CALL_RATIO', 'INSIDER_BUYING', 'NEWS_SENTIMENT', 'ANALYST_UPGRADES'],
            'volatility': ['REALIZED_VOL', 'IMPLIED_VOL', 'VOL_SKEW', 'GARCH_VOL', 'PARKINSON_VOL']
        }
        
        self.test_data['factor_categories'] = factor_categories
        
        # Generate synthetic factor values for testing
        factor_values = {}
        symbols = list(self.test_data['portfolio'].keys())[:10]  # Use first 10 symbols
        
        for symbol in symbols:
            factor_values[symbol] = {}
            for category, factors in factor_categories.items():
                for factor in factors:
                    # Generate realistic factor values
                    if factor.startswith('RSI'):
                        values = np.random.uniform(20, 80, 100).tolist()
                    elif factor.startswith('PE'):
                        values = np.random.exponential(15, 100).tolist()
                    elif factor.startswith('VOL'):
                        values = np.random.exponential(0.2, 100).tolist()
                    else:
                        values = np.random.normal(0, 1, 100).tolist()
                    
                    factor_values[symbol][f"{factor}_{category.upper()}"] = values
        
        self.test_data['factor_values'] = factor_values
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data for factor testing...")
        
        np.random.seed(42)
        
        # Generate portfolio data
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
        self.generate_factor_test_data()
    
    async def check_container_health(self) -> Dict[str, Any]:
        """Check if the Factor Engine container is healthy"""
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
        """Test the health endpoint and factor definitions status"""
        logger.info("Testing Factor Engine health endpoint...")
        
        results = {
            "test_name": "health_endpoint",
            "test_timestamp": datetime.now().isoformat(),
            "response_times": [],
            "success_rate": 0.0,
            "factor_definitions_loaded": 0,
            "cache_entries": 0,
            "queue_status": "unknown",
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
                                results["factor_definitions_loaded"] = data.get("factor_definitions_loaded", 0)
                                results["cache_entries"] = data.get("cache_entries", 0)
                                results["queue_size"] = data.get("queue_size", 0)
                                results["calculation_rate"] = data.get("calculation_rate", 0)
                                results["thread_pool_active"] = data.get("thread_pool_active", False)
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
    
    async def test_factor_calculation_performance(self) -> Dict[str, Any]:
        """Test factor calculation performance with varying numbers of factors"""
        logger.info("Testing factor calculation performance...")
        
        results = {
            "test_name": "factor_calculation_performance",
            "test_timestamp": datetime.now().isoformat(),
            "symbol_tests": [],
            "batch_tests": [],
            "throughput_factors_per_sec": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            # Test single symbol factor calculations with different factor counts
            symbols = list(self.test_data['portfolio'].keys())[:5]
            factor_counts = [1, 5, 10, 25, 50]
            
            for symbol in symbols[:3]:  # Test first 3 symbols
                for factor_count in factor_counts:
                    logger.info(f"Testing {symbol} with {factor_count} factors")
                    
                    # Create factor ID list (simulated)
                    factor_ids = [f"factor_{i}_{category}" for i in range(factor_count) 
                                 for category in ['technical', 'fundamental', 'sentiment']][:factor_count]
                    
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/factors/calculate/{symbol}",
                            json={"factor_ids": factor_ids} if factor_ids else None,
                            timeout=60
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status == 200:
                                data = await response.json()
                                symbol_result = {
                                    "symbol": symbol,
                                    "factor_count": factor_count,
                                    "success": True,
                                    "response_time_ms": response_time,
                                    "status": data.get("status", "unknown")
                                }
                                logger.info(f"Factor calc {symbol} ({factor_count}): {response_time:.2f}ms")
                            else:
                                symbol_result = {
                                    "symbol": symbol,
                                    "factor_count": factor_count,
                                    "success": False,
                                    "error": f"HTTP {response.status}"
                                }
                    
                    except Exception as e:
                        symbol_result = {
                            "symbol": symbol,
                            "factor_count": factor_count,
                            "success": False,
                            "error": str(e)
                        }
                    
                    results["symbol_tests"].append(symbol_result)
                    await asyncio.sleep(0.2)  # Brief delay
            
            # Test batch factor calculations (multiple symbols)
            batch_sizes = [2, 5, 10]
            
            for batch_size in batch_sizes:
                logger.info(f"Testing batch calculation with {batch_size} symbols")
                
                batch_symbols = symbols[:batch_size]
                batch_requests = []
                
                # Create concurrent requests for batch processing
                start_time = time.time()
                tasks = []
                
                for symbol in batch_symbols:
                    task = session.post(
                        f"{self.base_url}/factors/calculate/{symbol}",
                        timeout=60
                    )
                    tasks.append(task)
                
                try:
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    batch_time = (time.time() - start_time) * 1000
                    
                    successful_responses = 0
                    for response in responses:
                        if hasattr(response, 'status') and response.status == 200:
                            successful_responses += 1
                            response.close()
                        elif hasattr(response, 'close'):
                            response.close()
                    
                    batch_result = {
                        "batch_size": batch_size,
                        "successful_calculations": successful_responses,
                        "total_requests": len(batch_symbols),
                        "batch_time_ms": batch_time,
                        "success_rate": (successful_responses / len(batch_symbols)) * 100
                    }
                    
                    logger.info(f"Batch calc ({batch_size}): {batch_time:.2f}ms, {successful_responses}/{len(batch_symbols)} success")
                
                except Exception as e:
                    batch_result = {
                        "batch_size": batch_size,
                        "error": str(e),
                        "success": False
                    }
                
                results["batch_tests"].append(batch_result)
                await asyncio.sleep(1)  # Delay between batch tests
        
        return results
    
    async def test_factor_correlation_analysis(self) -> Dict[str, Any]:
        """Test factor correlation analysis capabilities"""
        logger.info("Testing factor correlation analysis...")
        
        results = {
            "test_name": "factor_correlation_analysis",
            "test_timestamp": datetime.now().isoformat(),
            "correlation_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test correlation analysis with different factor sets
            correlation_scenarios = [
                {
                    "name": "technical_factors",
                    "factors": ["RSI_14", "MACD_SIGNAL", "BB_WIDTH", "SMA_20", "EMA_12"],
                    "symbols": ["AAPL", "MSFT", "GOOGL"]
                },
                {
                    "name": "fundamental_factors", 
                    "factors": ["PE_RATIO", "PB_RATIO", "ROE", "DEBT_TO_EQUITY"],
                    "symbols": ["AAPL", "JPM", "JNJ"]
                },
                {
                    "name": "mixed_factors",
                    "factors": ["RSI_14", "PE_RATIO", "VIX_LEVEL", "GDP_GROWTH"],
                    "symbols": ["AAPL", "MSFT"]
                }
            ]
            
            for scenario in correlation_scenarios:
                logger.info(f"Testing correlation scenario: {scenario['name']}")
                
                correlation_request = {
                    "factors": scenario["factors"],
                    "symbols": scenario["symbols"],
                    "lookback_period": 252,  # 1 year
                    "correlation_method": "pearson",
                    "min_observations": 50
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/factors/correlations",
                        json=correlation_request,
                        timeout=60
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            correlation_result = {
                                "scenario_name": scenario["name"],
                                "success": True,
                                "response_time_ms": response_time,
                                "factor_count": len(scenario["factors"]),
                                "symbol_count": len(scenario["symbols"]),
                                "status": data.get("status", "unknown")
                            }
                            logger.info(f"Correlation {scenario['name']}: {response_time:.2f}ms")
                        else:
                            correlation_result = {
                                "scenario_name": scenario["name"],
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    correlation_result = {
                        "scenario_name": scenario["name"],
                        "success": False,
                        "error": str(e)
                    }
                
                results["correlation_tests"].append(correlation_result)
                await asyncio.sleep(0.5)
        
        return results
    
    async def test_high_volume_factor_processing(self) -> Dict[str, Any]:
        """Test high-volume factor processing capabilities"""
        logger.info("Testing high-volume factor processing...")
        
        results = {
            "test_name": "high_volume_processing",
            "test_timestamp": datetime.now().isoformat(),
            "volume_tests": [],
            "peak_throughput": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            # Test different volume levels
            volume_levels = [
                {"name": "medium_volume", "symbols": 5, "concurrent_requests": 3},
                {"name": "high_volume", "symbols": 10, "concurrent_requests": 5},
                {"name": "very_high_volume", "symbols": 15, "concurrent_requests": 8}
            ]
            
            for volume_config in volume_levels:
                logger.info(f"Testing volume level: {volume_config['name']}")
                
                symbols = list(self.test_data['portfolio'].keys())[:volume_config['symbols']]
                concurrent_requests = volume_config['concurrent_requests']
                
                # Create concurrent factor calculation requests
                start_time = time.time()
                tasks = []
                
                for i in range(concurrent_requests):
                    for symbol in symbols:
                        task = session.post(
                            f"{self.base_url}/factors/calculate/{symbol}",
                            timeout=120
                        )
                        tasks.append((symbol, task))
                
                try:
                    # Execute all requests concurrently
                    responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    total_time = (time.time() - start_time)
                    
                    successful_requests = 0
                    total_requests = len(tasks)
                    
                    for response in responses:
                        if hasattr(response, 'status') and response.status == 200:
                            successful_requests += 1
                            response.close()
                        elif hasattr(response, 'close'):
                            response.close()
                    
                    throughput = successful_requests / total_time if total_time > 0 else 0
                    
                    volume_result = {
                        "volume_level": volume_config['name'],
                        "total_requests": total_requests,
                        "successful_requests": successful_requests,
                        "total_time_seconds": total_time,
                        "throughput_requests_per_sec": throughput,
                        "success_rate": (successful_requests / total_requests) * 100,
                        "symbols_tested": len(symbols),
                        "concurrent_requests": concurrent_requests
                    }
                    
                    if throughput > results["peak_throughput"]:
                        results["peak_throughput"] = throughput
                    
                    logger.info(f"Volume {volume_config['name']}: {throughput:.2f} req/sec, "
                              f"{successful_requests}/{total_requests} success")
                
                except Exception as e:
                    volume_result = {
                        "volume_level": volume_config['name'],
                        "error": str(e),
                        "success": False
                    }
                
                results["volume_tests"].append(volume_result)
                await asyncio.sleep(2)  # Recovery time between volume tests
        
        return results
    
    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test the metrics endpoint for performance monitoring"""
        logger.info("Testing Factor Engine metrics endpoint...")
        
        results = {
            "test_name": "metrics_endpoint",
            "test_timestamp": datetime.now().isoformat(),
            "metrics_data": {},
            "response_time_ms": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.get(f"{self.base_url}/metrics", timeout=10) as response:
                    response_time = (time.time() - start_time) * 1000
                    results["response_time_ms"] = response_time
                    
                    if response.status == 200:
                        data = await response.json()
                        results["metrics_data"] = data
                        results["success"] = True
                        
                        logger.info(f"Metrics endpoint: {response_time:.2f}ms")
                        logger.info(f"Factors per second: {data.get('factors_per_second', 0):.2f}")
                        logger.info(f"Cache hit rate: {data.get('cache_hit_rate', 0):.2f}%")
                        logger.info(f"Active factor definitions: {data.get('active_factor_definitions', 0)}")
                    else:
                        results["success"] = False
                        results["error"] = f"HTTP {response.status}"
            
            except Exception as e:
                results["success"] = False
                results["error"] = str(e)
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all factor engine performance tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE FACTOR ENGINE PERFORMANCE TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "Factor Engine Comprehensive Performance Test",
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
            ("factor_calculation_performance", self.test_factor_calculation_performance),
            ("factor_correlation_analysis", self.test_factor_correlation_analysis),
            ("high_volume_processing", self.test_high_volume_factor_processing),
            ("metrics_endpoint", self.test_metrics_endpoint)
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
        logger.info("FACTOR ENGINE PERFORMANCE TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_performance_summary(self, results: Dict[str, Any]):
        """Create a performance summary from all test results"""
        summary = {
            "overall_health": "unknown",
            "factor_processing_performance": {},
            "system_metrics": {},
            "scalability_metrics": {}
        }
        
        # Health endpoint analysis
        health_test = results["test_results"].get("health_endpoint", {})
        if not health_test.get("test_failed"):
            summary["system_metrics"]["factor_definitions_loaded"] = health_test.get("factor_definitions_loaded", 0)
            summary["system_metrics"]["cache_entries"] = health_test.get("cache_entries", 0)
            summary["system_metrics"]["calculation_rate"] = health_test.get("calculation_rate", 0)
            summary["system_metrics"]["thread_pool_active"] = health_test.get("thread_pool_active", False)
        
        # Factor calculation performance
        calc_test = results["test_results"].get("factor_calculation_performance", {})
        if not calc_test.get("test_failed"):
            symbol_tests = calc_test.get("symbol_tests", [])
            successful_tests = [t for t in symbol_tests if t.get("success")]
            
            if successful_tests:
                avg_response_time = statistics.mean([t["response_time_ms"] for t in successful_tests])
                summary["factor_processing_performance"]["avg_calculation_time_ms"] = avg_response_time
                summary["factor_processing_performance"]["calculation_success_rate"] = (len(successful_tests) / len(symbol_tests)) * 100 if symbol_tests else 0
        
        # High volume processing
        volume_test = results["test_results"].get("high_volume_processing", {})
        if not volume_test.get("test_failed"):
            summary["scalability_metrics"]["peak_throughput_req_per_sec"] = volume_test.get("peak_throughput", 0)
            
            volume_tests = volume_test.get("volume_tests", [])
            if volume_tests:
                avg_success_rate = statistics.mean([t.get("success_rate", 0) for t in volume_tests if "success_rate" in t])
                summary["scalability_metrics"]["high_volume_success_rate"] = avg_success_rate
        
        # Metrics endpoint analysis
        metrics_test = results["test_results"].get("metrics_endpoint", {})
        if not metrics_test.get("test_failed") and metrics_test.get("success"):
            metrics_data = metrics_test.get("metrics_data", {})
            summary["system_metrics"]["factors_per_second"] = metrics_data.get("factors_per_second", 0)
            summary["system_metrics"]["cache_hit_rate"] = metrics_data.get("cache_hit_rate", 0)
            summary["system_metrics"]["active_factor_definitions"] = metrics_data.get("active_factor_definitions", 0)
        
        # Overall health assessment
        health_indicators = []
        
        if summary["system_metrics"].get("factor_definitions_loaded", 0) > 400:  # Expected ~485
            health_indicators.append(1)
        if summary["factor_processing_performance"].get("calculation_success_rate", 0) >= 90:
            health_indicators.append(1)
        if summary["scalability_metrics"].get("peak_throughput_req_per_sec", 0) > 1:
            health_indicators.append(1)
        if summary["system_metrics"].get("thread_pool_active"):
            health_indicators.append(1)
        
        if len(health_indicators) >= 3:
            summary["overall_health"] = "excellent"
        elif len(health_indicators) >= 2:
            summary["overall_health"] = "good"
        elif len(health_indicators) >= 1:
            summary["overall_health"] = "fair"
        else:
            summary["overall_health"] = "poor"
        
        results["performance_summary"] = summary
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate performance recommendations based on test results"""
        recommendations = []
        
        summary = results.get("performance_summary", {})
        
        # Factor definitions recommendations
        factor_definitions = summary.get("system_metrics", {}).get("factor_definitions_loaded", 0)
        if factor_definitions < 400:
            recommendations.append({
                "category": "factor_definitions",
                "severity": "high",
                "issue": f"Only {factor_definitions} factor definitions loaded (expected ~485)",
                "recommendation": "Check factor definition loading process and ensure all definitions are properly initialized"
            })
        
        # Performance recommendations
        calc_success_rate = summary.get("factor_processing_performance", {}).get("calculation_success_rate", 0)
        if calc_success_rate < 90:
            recommendations.append({
                "category": "calculation_performance",
                "severity": "high",
                "issue": f"Factor calculation success rate is {calc_success_rate:.1f}%",
                "recommendation": "Investigate factor calculation failures and optimize error handling"
            })
        
        # Throughput recommendations
        peak_throughput = summary.get("scalability_metrics", {}).get("peak_throughput_req_per_sec", 0)
        if peak_throughput < 5:
            recommendations.append({
                "category": "throughput",
                "severity": "medium",
                "issue": f"Peak throughput is only {peak_throughput:.1f} requests/second",
                "recommendation": "Consider increasing thread pool size or optimizing factor calculation algorithms"
            })
        
        # Cache performance recommendations
        cache_hit_rate = summary.get("system_metrics", {}).get("cache_hit_rate", 0)
        if cache_hit_rate < 70:
            recommendations.append({
                "category": "caching",
                "severity": "medium",
                "issue": f"Cache hit rate is {cache_hit_rate:.1f}%",
                "recommendation": "Optimize caching strategy and consider increasing cache size or improving cache eviction policies"
            })
        
        # Thread pool recommendations
        thread_pool_active = summary.get("system_metrics", {}).get("thread_pool_active", False)
        if not thread_pool_active:
            recommendations.append({
                "category": "threading",
                "severity": "high",
                "issue": "Thread pool is not active",
                "recommendation": "Investigate thread pool initialization and ensure proper resource allocation for parallel processing"
            })
        
        if not recommendations:
            recommendations.append({
                "category": "general",
                "severity": "info",
                "issue": "No critical issues detected",
                "recommendation": "Factor Engine is performing well. Continue monitoring factor calculations and throughput metrics."
            })
        
        results["recommendations"] = recommendations

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/factor_engine_performance_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = FactorEnginePerformanceTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("FACTOR ENGINE PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
    
    system_metrics = summary.get("system_metrics", {})
    if system_metrics:
        print(f"\nSystem Metrics:")
        print(f"  Factor Definitions Loaded: {system_metrics.get('factor_definitions_loaded', 0)}")
        print(f"  Cache Entries: {system_metrics.get('cache_entries', 0)}")
        print(f"  Factors per Second: {system_metrics.get('factors_per_second', 0):.2f}")
        print(f"  Cache Hit Rate: {system_metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"  Thread Pool Active: {system_metrics.get('thread_pool_active', False)}")
    
    processing_perf = summary.get("factor_processing_performance", {})
    if processing_perf:
        print(f"\nProcessing Performance:")
        print(f"  Average Calculation Time: {processing_perf.get('avg_calculation_time_ms', 0):.1f}ms")
        print(f"  Calculation Success Rate: {processing_perf.get('calculation_success_rate', 0):.1f}%")
    
    scalability = summary.get("scalability_metrics", {})
    if scalability:
        print(f"\nScalability Metrics:")
        print(f"  Peak Throughput: {scalability.get('peak_throughput_req_per_sec', 0):.1f} req/sec")
        print(f"  High Volume Success Rate: {scalability.get('high_volume_success_rate', 0):.1f}%")
    
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            severity = rec.get("severity", "info").upper()
            print(f"  {i}. [{severity}] {rec.get('recommendation', 'N/A')}")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
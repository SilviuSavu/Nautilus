#!/usr/bin/env python3
"""
Remaining Engines Comprehensive Performance Test
Tests the remaining containerized engines: Features, WebSocket, Strategy, Market Data, and Portfolio
"""
import asyncio
import json
import logging
import time
import statistics
import websockets
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

class RemainingEnginesPerformanceTest:
    """
    Comprehensive performance testing suite for remaining engines:
    - Features Engine (8500)
    - WebSocket Engine (8600) 
    - Strategy Engine (8700)
    - Market Data Engine (8800)
    - Portfolio Engine (8900)
    """
    
    def __init__(self):
        self.engines = {
            "features": {"url": "http://localhost:8500", "container": "nautilus-features-engine"},
            "websocket": {"url": "http://localhost:8600", "container": "nautilus-websocket-engine"},
            "strategy": {"url": "http://localhost:8700", "container": "nautilus-strategy-engine"},
            "marketdata": {"url": "http://localhost:8800", "container": "nautilus-marketdata-engine"},
            "portfolio": {"url": "http://localhost:8900", "container": "nautilus-portfolio-engine"}
        }
        
        self.docker_client = docker.from_env()
        self.test_data = {}
        self.performance_metrics = {}
        self.load_test_data()
        
    def load_test_data(self):
        """Load comprehensive test data for engine testing"""
        logger.info("Loading test data for remaining engines performance testing...")
        
        try:
            # Load portfolio data
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
            self.generate_engine_test_data()
            
            logger.info(f"Loaded test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.generate_synthetic_test_data()
    
    def generate_engine_test_data(self):
        """Generate specific test data for each engine type"""
        np.random.seed(42)
        
        # Features Engine test data
        self.test_data['features'] = {
            'technical_indicators': ['RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'SMA_20', 'EMA_12', 'STOCH_K', 'ADX'],
            'fundamental_metrics': ['PE_RATIO', 'PB_RATIO', 'ROE', 'DEBT_TO_EQUITY', 'CURRENT_RATIO'],
            'price_data': self.generate_price_series(['AAPL', 'MSFT', 'GOOGL'])
        }
        
        # Strategy Engine test data
        self.test_data['strategies'] = [
            {
                'strategy_id': 'momentum_strategy_v1',
                'strategy_type': 'momentum',
                'parameters': {'lookback': 20, 'threshold': 0.02},
                'symbols': ['AAPL', 'MSFT'],
                'risk_limits': {'max_position_size': 10000, 'stop_loss': 0.05}
            },
            {
                'strategy_id': 'mean_reversion_v2', 
                'strategy_type': 'mean_reversion',
                'parameters': {'window': 50, 'z_score_threshold': 2.0},
                'symbols': ['GOOGL'],
                'risk_limits': {'max_position_size': 15000, 'stop_loss': 0.03}
            }
        ]
        
        # Portfolio Engine test data
        self.test_data['portfolios'] = {
            'balanced_portfolio': {
                'assets': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                'weights': [0.25, 0.25, 0.25, 0.25],
                'optimization_method': 'mean_variance',
                'constraints': {'min_weight': 0.05, 'max_weight': 0.5}
            },
            'growth_portfolio': {
                'assets': ['TSLA', 'NVDA', 'AAPL', 'GOOGL'],
                'weights': [0.3, 0.3, 0.2, 0.2],
                'optimization_method': 'risk_parity',
                'constraints': {'min_weight': 0.1, 'max_weight': 0.4}
            }
        }
        
        # Market Data streaming test data
        self.test_data['market_data'] = self.generate_streaming_data()
    
    def generate_price_series(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Generate price series for technical indicator calculations"""
        price_series = {}
        
        for symbol in symbols:
            # Generate 100 days of OHLCV data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            base_price = np.random.uniform(100, 300)
            
            prices = []
            current_price = base_price
            
            for date in dates:
                # Generate daily price movement
                daily_return = np.random.normal(0.001, 0.02)
                current_price *= (1 + daily_return)
                
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                volume = int(np.random.exponential(1000000))
                
                prices.append({
                    'timestamp': date.isoformat(),
                    'open': round(current_price * (1 + np.random.normal(0, 0.001)), 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(current_price, 2),
                    'volume': volume
                })
            
            price_series[symbol] = prices
        
        return price_series
    
    def generate_streaming_data(self) -> List[Dict]:
        """Generate streaming market data for testing"""
        streaming_data = []
        base_time = datetime.now() - timedelta(minutes=10)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        base_prices = {symbol: np.random.uniform(100, 300) for symbol in symbols}
        
        for i in range(600):  # 10 minutes of data, 1 second intervals
            timestamp = base_time + timedelta(seconds=i)
            
            for symbol in symbols:
                # Generate realistic tick data
                price_change = np.random.normal(0, 0.001)
                new_price = base_prices[symbol] * (1 + price_change)
                base_prices[symbol] = new_price
                
                streaming_data.append({
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'price': round(new_price, 2),
                    'volume': int(np.random.exponential(1000)),
                    'bid': round(new_price - 0.01, 2),
                    'ask': round(new_price + 0.01, 2)
                })
        
        return streaming_data
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data for remaining engines...")
        
        np.random.seed(42)
        
        # Generate basic portfolio data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        portfolio_data = {}
        
        for symbol in symbols:
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = prices[1:]
            
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
        self.generate_engine_test_data()
    
    async def check_container_health(self, engine_name: str) -> Dict[str, Any]:
        """Check if a specific engine container is healthy"""
        try:
            container = self.docker_client.containers.get(self.engines[engine_name]["container"])
            
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
                "engine": engine_name,
                "container_id": container.id[:12],
                "status": container.status,
                "cpu_percent": round(cpu_percent, 2),
                "memory_usage_mb": round(memory_usage_mb, 2),
                "memory_limit_mb": round(memory_limit_mb, 2),
                "memory_percent": round((memory_usage_mb / memory_limit_mb) * 100, 2) if memory_limit_mb > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error checking container health for {engine_name}: {e}")
            return {"engine": engine_name, "error": str(e)}
    
    async def test_engine_health_endpoints(self) -> Dict[str, Any]:
        """Test health endpoints for all remaining engines"""
        logger.info("Testing health endpoints for all remaining engines...")
        
        results = {
            "test_name": "engine_health_endpoints",
            "test_timestamp": datetime.now().isoformat(),
            "engine_results": {},
            "overall_health": {}
        }
        
        async with aiohttp.ClientSession() as session:
            for engine_name, config in self.engines.items():
                logger.info(f"Testing {engine_name} engine health...")
                
                engine_results = {
                    "engine_name": engine_name,
                    "response_times": [],
                    "success_rate": 0.0,
                    "container_health": {}
                }
                
                # Test health endpoint multiple times
                for i in range(5):
                    start_time = time.time()
                    try:
                        async with session.get(f"{config['url']}/health", timeout=10) as response:
                            response_time = (time.time() - start_time) * 1000
                            engine_results["response_times"].append(response_time)
                            
                            if response.status == 200:
                                data = await response.json()
                                if i == 0:  # First successful response
                                    engine_results["health_data"] = data
                                logger.info(f"{engine_name} health {i+1}: {response_time:.2f}ms - {data.get('status', 'unknown')}")
                            else:
                                logger.warning(f"{engine_name} health {i+1}: HTTP {response.status}")
                    
                    except Exception as e:
                        logger.error(f"{engine_name} health {i+1} failed: {e}")
                    
                    await asyncio.sleep(0.1)
                
                # Calculate statistics
                if engine_results["response_times"]:
                    engine_results["success_rate"] = len(engine_results["response_times"]) / 5 * 100
                    engine_results["avg_response_time"] = statistics.mean(engine_results["response_times"])
                
                # Get container health
                engine_results["container_health"] = await self.check_container_health(engine_name)
                
                results["engine_results"][engine_name] = engine_results
        
        # Calculate overall health summary
        success_rates = [r.get("success_rate", 0) for r in results["engine_results"].values()]
        if success_rates:
            results["overall_health"]["avg_success_rate"] = statistics.mean(success_rates)
            results["overall_health"]["engines_healthy"] = len([r for r in success_rates if r >= 80])
            results["overall_health"]["total_engines"] = len(success_rates)
        
        return results
    
    async def test_features_engine_performance(self) -> Dict[str, Any]:
        """Test Features Engine performance with technical indicators"""
        logger.info("Testing Features Engine performance...")
        
        results = {
            "test_name": "features_engine_performance",
            "test_timestamp": datetime.now().isoformat(),
            "indicator_tests": [],
            "batch_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test individual technical indicators
            indicators = self.test_data['features']['technical_indicators'][:5]  # Test first 5
            
            for indicator in indicators:
                for symbol in ['AAPL', 'MSFT']:
                    price_data = self.test_data['features']['price_data'].get(symbol, [])[:50]  # 50 days
                    
                    indicator_request = {
                        "symbol": symbol,
                        "indicator": indicator,
                        "price_data": price_data,
                        "parameters": self.get_indicator_parameters(indicator)
                    }
                    
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.engines['features']['url']}/features/calculate/{indicator}",
                            json=indicator_request,
                            timeout=30
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            if response.status == 200:
                                data = await response.json()
                                indicator_result = {
                                    "indicator": indicator,
                                    "symbol": symbol,
                                    "success": True,
                                    "response_time_ms": response_time,
                                    "values_calculated": len(data.get("values", [])) if isinstance(data.get("values"), list) else 1
                                }
                                logger.info(f"Features {indicator} for {symbol}: {response_time:.2f}ms")
                            else:
                                indicator_result = {
                                    "indicator": indicator,
                                    "symbol": symbol,
                                    "success": False,
                                    "error": f"HTTP {response.status}"
                                }
                    
                    except Exception as e:
                        indicator_result = {
                            "indicator": indicator,
                            "symbol": symbol,
                            "success": False,
                            "error": str(e)
                        }
                    
                    results["indicator_tests"].append(indicator_result)
                    await asyncio.sleep(0.1)
            
            # Test batch indicator calculation
            batch_request = {
                "symbols": ["AAPL", "MSFT"],
                "indicators": indicators[:3],  # First 3 indicators
                "price_data": {
                    symbol: self.test_data['features']['price_data'].get(symbol, [])[:30] 
                    for symbol in ["AAPL", "MSFT"]
                }
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['features']['url']}/features/batch",
                    json=batch_request,
                    timeout=60
                ) as response:
                    batch_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        batch_result = {
                            "success": True,
                            "response_time_ms": batch_time,
                            "symbols_processed": len(batch_request["symbols"]),
                            "indicators_processed": len(batch_request["indicators"]),
                            "total_calculations": len(batch_request["symbols"]) * len(batch_request["indicators"])
                        }
                        logger.info(f"Features batch processing: {batch_time:.2f}ms, {batch_result['total_calculations']} calculations")
                    else:
                        batch_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                batch_result = {"success": False, "error": str(e)}
            
            results["batch_tests"].append(batch_result)
        
        return results
    
    def get_indicator_parameters(self, indicator: str) -> Dict[str, Any]:
        """Get default parameters for technical indicators"""
        params = {
            'RSI': {'period': 14},
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'BB_UPPER': {'period': 20, 'std_dev': 2},
            'BB_LOWER': {'period': 20, 'std_dev': 2},
            'SMA_20': {'period': 20},
            'EMA_12': {'period': 12},
            'STOCH_K': {'k_period': 14, 'd_period': 3},
            'ADX': {'period': 14}
        }
        return params.get(indicator, {})
    
    async def test_websocket_engine_performance(self) -> Dict[str, Any]:
        """Test WebSocket Engine performance and connection handling"""
        logger.info("Testing WebSocket Engine performance...")
        
        results = {
            "test_name": "websocket_engine_performance",
            "test_timestamp": datetime.now().isoformat(),
            "connection_tests": [],
            "streaming_tests": [],
            "concurrent_connection_test": {}
        }
        
        # Test WebSocket connections through HTTP API first
        async with aiohttp.ClientSession() as session:
            # Test WebSocket endpoint status
            try:
                async with session.get(f"{self.engines['websocket']['url']}/websocket/status", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results["websocket_status"] = data
                        logger.info(f"WebSocket status: {data.get('active_connections', 0)} connections")
            except Exception as e:
                results["websocket_status"] = {"error": str(e)}
            
            # Test subscription management
            subscription_request = {
                "client_id": "test_client_1",
                "subscriptions": ["market_data", "portfolio_updates", "risk_alerts"],
                "symbols": ["AAPL", "MSFT"]
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['websocket']['url']}/websocket/subscribe",
                    json=subscription_request,
                    timeout=10
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        subscription_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "subscriptions_created": len(subscription_request["subscriptions"])
                        }
                    else:
                        subscription_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                subscription_result = {"success": False, "error": str(e)}
            
            results["connection_tests"].append(subscription_result)
        
        # Test message broadcasting
        broadcast_test = await self.test_websocket_broadcasting()
        results["streaming_tests"].append(broadcast_test)
        
        return results
    
    async def test_websocket_broadcasting(self) -> Dict[str, Any]:
        """Test WebSocket message broadcasting performance"""
        async with aiohttp.ClientSession() as session:
            # Test broadcasting market data
            market_data = self.test_data['market_data'][:10]  # First 10 ticks
            
            broadcast_request = {
                "topic": "market_data",
                "data": market_data,
                "target_clients": ["all"]
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['websocket']['url']}/websocket/broadcast",
                    json=broadcast_request,
                    timeout=30
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "test_name": "websocket_broadcasting",
                            "success": True,
                            "response_time_ms": response_time,
                            "messages_broadcast": len(market_data),
                            "clients_reached": data.get("clients_reached", 0)
                        }
                    else:
                        return {"test_name": "websocket_broadcasting", "success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                return {"test_name": "websocket_broadcasting", "success": False, "error": str(e)}
    
    async def test_strategy_engine_performance(self) -> Dict[str, Any]:
        """Test Strategy Engine performance with strategy deployment and execution"""
        logger.info("Testing Strategy Engine performance...")
        
        results = {
            "test_name": "strategy_engine_performance",
            "test_timestamp": datetime.now().isoformat(),
            "deployment_tests": [],
            "execution_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test strategy deployment
            for strategy in self.test_data['strategies']:
                logger.info(f"Testing strategy deployment: {strategy['strategy_id']}")
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.engines['strategy']['url']}/strategy/deploy",
                        json=strategy,
                        timeout=60
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            deployment_result = {
                                "strategy_id": strategy["strategy_id"],
                                "success": True,
                                "response_time_ms": response_time,
                                "deployment_status": data.get("status", "unknown")
                            }
                            logger.info(f"Strategy deployment {strategy['strategy_id']}: {response_time:.2f}ms")
                        else:
                            deployment_result = {
                                "strategy_id": strategy["strategy_id"],
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    deployment_result = {
                        "strategy_id": strategy["strategy_id"],
                        "success": False,
                        "error": str(e)
                    }
                
                results["deployment_tests"].append(deployment_result)
                await asyncio.sleep(0.5)
            
            # Test strategy execution
            execution_request = {
                "strategy_id": self.test_data['strategies'][0]['strategy_id'],
                "market_data": self.test_data['market_data'][:50],  # 50 data points
                "execution_mode": "backtest"
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['strategy']['url']}/strategy/execute",
                    json=execution_request,
                    timeout=120
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        execution_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "trades_generated": data.get("trades_generated", 0),
                            "performance_metrics": data.get("performance", {})
                        }
                        logger.info(f"Strategy execution: {response_time:.2f}ms, {execution_result['trades_generated']} trades")
                    else:
                        execution_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                execution_result = {"success": False, "error": str(e)}
            
            results["execution_tests"].append(execution_result)
        
        return results
    
    async def test_marketdata_engine_performance(self) -> Dict[str, Any]:
        """Test Market Data Engine performance with data ingestion and distribution"""
        logger.info("Testing Market Data Engine performance...")
        
        results = {
            "test_name": "marketdata_engine_performance",
            "test_timestamp": datetime.now().isoformat(),
            "ingestion_tests": [],
            "distribution_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test market data ingestion
            batch_sizes = [10, 50, 100]
            
            for batch_size in batch_sizes:
                logger.info(f"Testing market data ingestion with batch size: {batch_size}")
                
                market_data_batch = self.test_data['market_data'][:batch_size]
                
                ingestion_request = {
                    "source": "test_feed",
                    "data_type": "tick_data",
                    "market_data": market_data_batch
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.engines['marketdata']['url']}/marketdata/ingest",
                        json=ingestion_request,
                        timeout=60
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            ingestion_result = {
                                "batch_size": batch_size,
                                "success": True,
                                "response_time_ms": response_time,
                                "throughput_msgs_per_sec": batch_size / (response_time / 1000) if response_time > 0 else 0,
                                "records_processed": data.get("records_processed", 0)
                            }
                            logger.info(f"Market data ingestion ({batch_size}): {response_time:.2f}ms, "
                                      f"{ingestion_result['throughput_msgs_per_sec']:.1f} msgs/sec")
                        else:
                            ingestion_result = {
                                "batch_size": batch_size,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    ingestion_result = {
                        "batch_size": batch_size,
                        "success": False,
                        "error": str(e)
                    }
                
                results["ingestion_tests"].append(ingestion_result)
                await asyncio.sleep(0.5)
            
            # Test data distribution
            distribution_request = {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "data_types": ["quotes", "trades", "bars"],
                "subscribers": ["analytics_engine", "risk_engine"]
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['marketdata']['url']}/marketdata/distribute",
                    json=distribution_request,
                    timeout=30
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        distribution_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "symbols_distributed": len(distribution_request["symbols"]),
                            "subscribers_notified": data.get("subscribers_notified", 0)
                        }
                        logger.info(f"Market data distribution: {response_time:.2f}ms")
                    else:
                        distribution_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                distribution_result = {"success": False, "error": str(e)}
            
            results["distribution_tests"].append(distribution_result)
        
        return results
    
    async def test_portfolio_engine_performance(self) -> Dict[str, Any]:
        """Test Portfolio Engine performance with optimization and analysis"""
        logger.info("Testing Portfolio Engine performance...")
        
        results = {
            "test_name": "portfolio_engine_performance",
            "test_timestamp": datetime.now().isoformat(),
            "optimization_tests": [],
            "analysis_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test portfolio optimization
            for portfolio_name, portfolio_config in self.test_data['portfolios'].items():
                logger.info(f"Testing portfolio optimization: {portfolio_name}")
                
                # Add historical returns data for optimization
                returns_data = {}
                for asset in portfolio_config['assets']:
                    if asset in self.test_data['portfolio']:
                        price_data = self.test_data['portfolio'][asset]
                        dates = sorted(price_data.keys())[-100:]  # Last 100 days
                        prices = [price_data[date]['Close'] for date in dates]
                        returns = np.diff(prices) / prices[:-1]
                        returns_data[asset] = returns.tolist()
                
                optimization_request = {
                    **portfolio_config,
                    "historical_returns": returns_data,
                    "risk_free_rate": 0.02,
                    "optimization_period": "1Y"
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.engines['portfolio']['url']}/portfolio/optimize",
                        json=optimization_request,
                        timeout=180  # Portfolio optimization can take time
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            optimization_result = {
                                "portfolio_name": portfolio_name,
                                "success": True,
                                "response_time_ms": response_time,
                                "method": portfolio_config["optimization_method"],
                                "assets_count": len(portfolio_config["assets"]),
                                "optimal_weights": data.get("optimal_weights", {}),
                                "expected_return": data.get("expected_return", 0),
                                "portfolio_risk": data.get("portfolio_risk", 0)
                            }
                            logger.info(f"Portfolio optimization {portfolio_name}: {response_time:.2f}ms")
                        else:
                            optimization_result = {
                                "portfolio_name": portfolio_name,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    optimization_result = {
                        "portfolio_name": portfolio_name,
                        "success": False,
                        "error": str(e)
                    }
                
                results["optimization_tests"].append(optimization_result)
                await asyncio.sleep(1)
            
            # Test portfolio analysis
            analysis_request = {
                "portfolio_id": "test_portfolio_analysis",
                "assets": ["AAPL", "MSFT", "GOOGL"],
                "weights": [0.4, 0.3, 0.3],
                "benchmark": "SPY",
                "analysis_period": "1Y"
            }
            
            start_time = time.time()
            try:
                async with session.post(
                    f"{self.engines['portfolio']['url']}/portfolio/analyze",
                    json=analysis_request,
                    timeout=120
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        analysis_result = {
                            "success": True,
                            "response_time_ms": response_time,
                            "metrics_calculated": len(data.get("metrics", {})),
                            "risk_metrics": data.get("risk_metrics", {}),
                            "performance_metrics": data.get("performance_metrics", {})
                        }
                        logger.info(f"Portfolio analysis: {response_time:.2f}ms, {analysis_result['metrics_calculated']} metrics")
                    else:
                        analysis_result = {"success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                analysis_result = {"success": False, "error": str(e)}
            
            results["analysis_tests"].append(analysis_result)
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all remaining engines performance tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE REMAINING ENGINES PERFORMANCE TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "Remaining Engines Comprehensive Performance Test",
            "test_start_time": datetime.now().isoformat(),
            "engines_tested": list(self.engines.keys()),
            "test_results": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Test sequence
        test_sequence = [
            ("engine_health_endpoints", self.test_engine_health_endpoints),
            ("features_engine_performance", self.test_features_engine_performance),
            ("websocket_engine_performance", self.test_websocket_engine_performance),
            ("strategy_engine_performance", self.test_strategy_engine_performance),
            ("marketdata_engine_performance", self.test_marketdata_engine_performance),
            ("portfolio_engine_performance", self.test_portfolio_engine_performance)
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
        logger.info("REMAINING ENGINES PERFORMANCE TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_performance_summary(self, results: Dict[str, Any]):
        """Create a performance summary from all test results"""
        summary = {
            "overall_health": "unknown",
            "engine_health_summary": {},
            "individual_engine_performance": {},
            "system_integration": {}
        }
        
        # Engine health analysis
        health_test = results["test_results"].get("engine_health_endpoints", {})
        if not health_test.get("test_failed"):
            overall_health = health_test.get("overall_health", {})
            summary["engine_health_summary"] = overall_health
            
            # Individual engine health
            for engine_name, engine_result in health_test.get("engine_results", {}).items():
                summary["individual_engine_performance"][engine_name] = {
                    "success_rate": engine_result.get("success_rate", 0),
                    "avg_response_time": engine_result.get("avg_response_time", 0),
                    "container_status": engine_result.get("container_health", {}).get("status", "unknown")
                }
        
        # Specific engine performance analysis
        performance_tests = [
            ("features", "features_engine_performance"),
            ("websocket", "websocket_engine_performance"), 
            ("strategy", "strategy_engine_performance"),
            ("marketdata", "marketdata_engine_performance"),
            ("portfolio", "portfolio_engine_performance")
        ]
        
        for engine_name, test_name in performance_tests:
            test_result = results["test_results"].get(test_name, {})
            if not test_result.get("test_failed"):
                if engine_name not in summary["individual_engine_performance"]:
                    summary["individual_engine_performance"][engine_name] = {}
                
                # Extract key performance metrics based on engine type
                if engine_name == "features":
                    indicator_tests = test_result.get("indicator_tests", [])
                    successful_tests = [t for t in indicator_tests if t.get("success")]
                    summary["individual_engine_performance"][engine_name]["indicator_success_rate"] = (len(successful_tests) / len(indicator_tests)) * 100 if indicator_tests else 0
                
                elif engine_name == "marketdata":
                    ingestion_tests = test_result.get("ingestion_tests", [])
                    if ingestion_tests:
                        throughputs = [t.get("throughput_msgs_per_sec", 0) for t in ingestion_tests if t.get("success")]
                        summary["individual_engine_performance"][engine_name]["peak_throughput_msgs_per_sec"] = max(throughputs) if throughputs else 0
                
                elif engine_name == "portfolio":
                    optimization_tests = test_result.get("optimization_tests", [])
                    successful_optimizations = [t for t in optimization_tests if t.get("success")]
                    summary["individual_engine_performance"][engine_name]["optimization_success_rate"] = (len(successful_optimizations) / len(optimization_tests)) * 100 if optimization_tests else 0
        
        # Overall health assessment
        engine_success_rates = [perf.get("success_rate", 0) for perf in summary["individual_engine_performance"].values()]
        
        if engine_success_rates:
            avg_success_rate = statistics.mean(engine_success_rates)
            healthy_engines = len([rate for rate in engine_success_rates if rate >= 80])
            total_engines = len(engine_success_rates)
            
            if healthy_engines == total_engines and avg_success_rate >= 90:
                summary["overall_health"] = "excellent"
            elif healthy_engines >= total_engines * 0.8 and avg_success_rate >= 80:
                summary["overall_health"] = "good"
            elif healthy_engines >= total_engines * 0.6:
                summary["overall_health"] = "fair"
            else:
                summary["overall_health"] = "poor"
        
        results["performance_summary"] = summary

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/remaining_engines_performance_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = RemainingEnginesPerformanceTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("REMAINING ENGINES PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
    
    engine_health = summary.get("engine_health_summary", {})
    if engine_health:
        print(f"\nEngine Health Summary:")
        print(f"  Average Success Rate: {engine_health.get('avg_success_rate', 0):.1f}%")
        print(f"  Healthy Engines: {engine_health.get('engines_healthy', 0)}/{engine_health.get('total_engines', 0)}")
    
    individual_perf = summary.get("individual_engine_performance", {})
    if individual_perf:
        print(f"\nIndividual Engine Performance:")
        for engine_name, perf in individual_perf.items():
            print(f"  {engine_name.upper()}:")
            print(f"    Success Rate: {perf.get('success_rate', 0):.1f}%")
            print(f"    Avg Response Time: {perf.get('avg_response_time', 0):.1f}ms")
            print(f"    Container Status: {perf.get('container_status', 'unknown')}")
            
            # Engine-specific metrics
            if engine_name == "features":
                print(f"    Indicator Success Rate: {perf.get('indicator_success_rate', 0):.1f}%")
            elif engine_name == "marketdata":
                print(f"    Peak Throughput: {perf.get('peak_throughput_msgs_per_sec', 0):.1f} msgs/sec")
            elif engine_name == "portfolio":
                print(f"    Optimization Success Rate: {perf.get('optimization_success_rate', 0):.1f}%")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
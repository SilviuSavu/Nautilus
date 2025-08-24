#!/usr/bin/env python3
"""
Inter-Engine Integration Performance Test
Tests communication, data flow, and integration between all containerized engines
Validates MessageBus integration, event-driven architecture, and system coordination
"""
import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import pandas as pd
import numpy as np
import psutil
import docker
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterEngineIntegrationTest:
    """
    Comprehensive integration testing suite for inter-engine communication
    Tests MessageBus integration, event flow, and system coordination across all 9 engines
    """
    
    def __init__(self):
        self.engines = {
            "analytics": {"url": "http://localhost:8100", "container": "nautilus-analytics-engine"},
            "risk": {"url": "http://localhost:8200", "container": "nautilus-risk-engine"},
            "factor": {"url": "http://localhost:8300", "container": "nautilus-factor-engine"},
            "ml": {"url": "http://localhost:8400", "container": "nautilus-ml-engine"},
            "features": {"url": "http://localhost:8500", "container": "nautilus-features-engine"},
            "websocket": {"url": "http://localhost:8600", "container": "nautilus-websocket-engine"},
            "strategy": {"url": "http://localhost:8700", "container": "nautilus-strategy-engine"},
            "marketdata": {"url": "http://localhost:8800", "container": "nautilus-marketdata-engine"},
            "portfolio": {"url": "http://localhost:8900", "container": "nautilus-portfolio-engine"}
        }
        
        self.docker_client = docker.from_env()
        self.test_data = {}
        self.performance_metrics = {}
        self.integration_flows = []
        self.load_test_data()
        self.define_integration_flows()
        
    def load_test_data(self):
        """Load comprehensive test data for integration testing"""
        logger.info("Loading test data for inter-engine integration testing...")
        
        try:
            # Load portfolio data
            portfolio_data = {}
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            for symbol in symbols:
                try:
                    with open(f'/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/yfinance_{symbol}_daily.json', 'r') as f:
                        data = json.load(f)
                        portfolio_data[symbol] = data
                except FileNotFoundError:
                    logger.warning(f"Could not load data for {symbol}")
            
            self.test_data['portfolio'] = portfolio_data
            self.generate_integration_test_data()
            
            logger.info(f"Loaded integration test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.generate_synthetic_test_data()
    
    def generate_integration_test_data(self):
        """Generate integration-specific test data"""
        np.random.seed(42)
        
        # Trading scenario data for integration testing
        self.test_data['trading_scenario'] = {
            'portfolio_id': 'integration_test_portfolio',
            'initial_capital': 1000000,
            'positions': {
                'AAPL': {'shares': 1000, 'entry_price': 150.0},
                'MSFT': {'shares': 800, 'entry_price': 250.0},
                'GOOGL': {'shares': 300, 'entry_price': 2500.0}
            },
            'risk_limits': {
                'max_portfolio_loss': 50000,
                'max_position_size': 100000,
                'var_limit': 75000
            }
        }
        
        # Market data for real-time simulation
        self.test_data['live_market_data'] = self.generate_live_market_data()
        
        # Strategy parameters for testing
        self.test_data['test_strategy'] = {
            'strategy_id': 'integration_test_momentum',
            'strategy_type': 'momentum',
            'parameters': {'lookback': 20, 'threshold': 0.02},
            'symbols': ['AAPL', 'MSFT'],
            'risk_limits': {'max_position_size': 50000}
        }
    
    def generate_live_market_data(self) -> List[Dict]:
        """Generate realistic live market data for integration testing"""
        market_data = []
        base_time = datetime.now()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        base_prices = {'AAPL': 150.0, 'MSFT': 250.0, 'GOOGL': 2500.0}
        
        for i in range(100):  # 100 ticks
            timestamp = base_time + timedelta(seconds=i * 3)
            
            for symbol in symbols:
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.002)  # 0.2% volatility
                new_price = base_prices[symbol] * (1 + price_change)
                base_prices[symbol] = new_price
                
                market_data.append({
                    'timestamp': timestamp.isoformat(),
                    'symbol': symbol,
                    'price': round(new_price, 2),
                    'volume': int(np.random.exponential(10000)),
                    'bid': round(new_price - 0.02, 2),
                    'ask': round(new_price + 0.02, 2),
                    'trade_id': f"trade_{i}_{symbol}"
                })
        
        return market_data
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data for integration testing...")
        
        np.random.seed(42)
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
                daily_data[date.isoformat()] = {
                    'Open': round(prices[i] * (1 + np.random.normal(0, 0.001)), 2),
                    'High': round(prices[i] * (1 + abs(np.random.normal(0, 0.005))), 2),
                    'Low': round(prices[i] * (1 - abs(np.random.normal(0, 0.005))), 2),
                    'Close': round(prices[i], 2),
                    'Volume': int(np.random.exponential(1000000)),
                    'Adj Close': round(prices[i], 2)
                }
            
            portfolio_data[symbol] = daily_data
        
        self.test_data['portfolio'] = portfolio_data
        self.generate_integration_test_data()
    
    def define_integration_flows(self):
        """Define key integration flows to test between engines"""
        self.integration_flows = [
            {
                'name': 'market_data_to_analytics_flow',
                'description': 'Market Data → Analytics → Risk',
                'sequence': ['marketdata', 'analytics', 'risk'],
                'data_type': 'market_data',
                'expected_latency_ms': 100
            },
            {
                'name': 'strategy_execution_flow',
                'description': 'Strategy → Risk → Portfolio → Analytics',
                'sequence': ['strategy', 'risk', 'portfolio', 'analytics'],
                'data_type': 'strategy_signal',
                'expected_latency_ms': 200
            },
            {
                'name': 'factor_to_ml_flow',
                'description': 'Factor → Features → ML → Strategy',
                'sequence': ['factor', 'features', 'ml', 'strategy'],
                'data_type': 'factor_data',
                'expected_latency_ms': 500
            },
            {
                'name': 'real_time_risk_flow',
                'description': 'Market Data → Risk → WebSocket',
                'sequence': ['marketdata', 'risk', 'websocket'],
                'data_type': 'risk_alert',
                'expected_latency_ms': 50
            },
            {
                'name': 'portfolio_optimization_flow',
                'description': 'Analytics → Risk → Portfolio → Strategy',
                'sequence': ['analytics', 'risk', 'portfolio', 'strategy'],
                'data_type': 'portfolio_rebalance',
                'expected_latency_ms': 1000
            }
        ]
    
    async def check_all_engines_health(self) -> Dict[str, Any]:
        """Check health status of all engines before integration testing"""
        logger.info("Checking health of all engines...")
        
        results = {
            "test_name": "all_engines_health_check",
            "test_timestamp": datetime.now().isoformat(),
            "engine_health": {},
            "healthy_engines": 0,
            "total_engines": len(self.engines)
        }
        
        async with aiohttp.ClientSession() as session:
            health_tasks = []
            
            for engine_name, config in self.engines.items():
                task = self.check_engine_health(session, engine_name, config['url'])
                health_tasks.append((engine_name, task))
            
            # Execute health checks concurrently
            for engine_name, task in health_tasks:
                try:
                    health_result = await task
                    results["engine_health"][engine_name] = health_result
                    
                    if health_result.get("status") == "healthy":
                        results["healthy_engines"] += 1
                        
                except Exception as e:
                    results["engine_health"][engine_name] = {"status": "error", "error": str(e)}
        
        results["health_percentage"] = (results["healthy_engines"] / results["total_engines"]) * 100
        
        logger.info(f"Engine health check complete: {results['healthy_engines']}/{results['total_engines']} engines healthy")
        
        return results
    
    async def check_engine_health(self, session: aiohttp.ClientSession, engine_name: str, url: str) -> Dict[str, Any]:
        """Check health of a specific engine"""
        try:
            async with session.get(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": data.get("status", "unknown"),
                        "response_time_ms": 0,  # Will be calculated by caller
                        "additional_info": {
                            "uptime": data.get("uptime_seconds", 0),
                            "processed_count": data.get("processed_count", data.get("predictions_made", data.get("factors_calculated", 0)))
                        }
                    }
                else:
                    return {"status": "unhealthy", "http_status": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def test_market_data_flow_integration(self) -> Dict[str, Any]:
        """Test market data flowing through multiple engines"""
        logger.info("Testing market data flow integration...")
        
        results = {
            "test_name": "market_data_flow_integration",
            "test_timestamp": datetime.now().isoformat(),
            "flow_tests": [],
            "end_to_end_latency": {}
        }
        
        async with aiohttp.ClientSession() as session:
            # Test Market Data → Analytics → Risk flow
            market_data_batch = self.test_data['live_market_data'][:10]
            
            # Step 1: Send market data to Market Data Engine
            step1_start = time.time()
            try:
                async with session.post(
                    f"{self.engines['marketdata']['url']}/marketdata/ingest",
                    json={
                        "source": "integration_test",
                        "data_type": "tick_data",
                        "market_data": market_data_batch
                    },
                    timeout=30
                ) as response:
                    step1_time = (time.time() - step1_start) * 1000
                    
                    if response.status == 200:
                        step1_result = {
                            "step": "market_data_ingestion",
                            "success": True,
                            "response_time_ms": step1_time,
                            "data_points": len(market_data_batch)
                        }
                    else:
                        step1_result = {"step": "market_data_ingestion", "success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                step1_result = {"step": "market_data_ingestion", "success": False, "error": str(e)}
            
            results["flow_tests"].append(step1_result)
            
            # Wait for data propagation
            await asyncio.sleep(2)
            
            # Step 2: Trigger Analytics calculation
            if step1_result.get("success"):
                step2_start = time.time()
                try:
                    async with session.post(
                        f"{self.engines['analytics']['url']}/analytics/performance/{self.test_data['trading_scenario']['portfolio_id']}",
                        json={
                            "positions": self.test_data['trading_scenario']['positions'],
                            "market_data_source": "integration_test"
                        },
                        timeout=30
                    ) as response:
                        step2_time = (time.time() - step2_start) * 1000
                        
                        if response.status == 200:
                            step2_result = {
                                "step": "analytics_calculation",
                                "success": True,
                                "response_time_ms": step2_time
                            }
                        else:
                            step2_result = {"step": "analytics_calculation", "success": False, "error": f"HTTP {response.status}"}
                
                except Exception as e:
                    step2_result = {"step": "analytics_calculation", "success": False, "error": str(e)}
                
                results["flow_tests"].append(step2_result)
                
                # Step 3: Trigger Risk check
                if step2_result.get("success"):
                    step3_start = time.time()
                    try:
                        async with session.post(
                            f"{self.engines['risk']['url']}/risk/check/{self.test_data['trading_scenario']['portfolio_id']}",
                            json={
                                "position_data": self.test_data['trading_scenario']['positions'],
                                "risk_limits": self.test_data['trading_scenario']['risk_limits']
                            },
                            timeout=30
                        ) as response:
                            step3_time = (time.time() - step3_start) * 1000
                            
                            if response.status == 200:
                                step3_result = {
                                    "step": "risk_check",
                                    "success": True,
                                    "response_time_ms": step3_time
                                }
                            else:
                                step3_result = {"step": "risk_check", "success": False, "error": f"HTTP {response.status}"}
                    
                    except Exception as e:
                        step3_result = {"step": "risk_check", "success": False, "error": str(e)}
                    
                    results["flow_tests"].append(step3_result)
            
            # Calculate end-to-end latency
            successful_steps = [step for step in results["flow_tests"] if step.get("success")]
            if successful_steps:
                total_latency = sum(step["response_time_ms"] for step in successful_steps)
                results["end_to_end_latency"]["market_data_flow"] = {
                    "total_latency_ms": total_latency,
                    "steps_completed": len(successful_steps),
                    "success_rate": (len(successful_steps) / len(results["flow_tests"])) * 100
                }
        
        return results
    
    async def test_strategy_execution_integration(self) -> Dict[str, Any]:
        """Test strategy execution across multiple engines"""
        logger.info("Testing strategy execution integration...")
        
        results = {
            "test_name": "strategy_execution_integration",
            "test_timestamp": datetime.now().isoformat(),
            "execution_steps": [],
            "strategy_performance": {}
        }
        
        async with aiohttp.ClientSession() as session:
            strategy = self.test_data['test_strategy']
            
            # Step 1: Deploy strategy
            step1_start = time.time()
            try:
                async with session.post(
                    f"{self.engines['strategy']['url']}/strategy/deploy",
                    json=strategy,
                    timeout=60
                ) as response:
                    step1_time = (time.time() - step1_start) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        step1_result = {
                            "step": "strategy_deployment",
                            "success": True,
                            "response_time_ms": step1_time,
                            "strategy_id": strategy["strategy_id"],
                            "deployment_status": data.get("status")
                        }
                    else:
                        step1_result = {"step": "strategy_deployment", "success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                step1_result = {"step": "strategy_deployment", "success": False, "error": str(e)}
            
            results["execution_steps"].append(step1_result)
            
            # Step 2: Risk validation
            if step1_result.get("success"):
                await asyncio.sleep(1)  # Allow deployment to complete
                
                step2_start = time.time()
                try:
                    async with session.post(
                        f"{self.engines['risk']['url']}/risk/check/strategy_{strategy['strategy_id']}",
                        json={
                            "strategy_config": strategy,
                            "position_limits": strategy["risk_limits"]
                        },
                        timeout=30
                    ) as response:
                        step2_time = (time.time() - step2_start) * 1000
                        
                        if response.status == 200:
                            step2_result = {
                                "step": "risk_validation",
                                "success": True,
                                "response_time_ms": step2_time
                            }
                        else:
                            step2_result = {"step": "risk_validation", "success": False, "error": f"HTTP {response.status}"}
                
                except Exception as e:
                    step2_result = {"step": "risk_validation", "success": False, "error": str(e)}
                
                results["execution_steps"].append(step2_result)
                
                # Step 3: Portfolio impact analysis
                if step2_result.get("success"):
                    step3_start = time.time()
                    try:
                        async with session.post(
                            f"{self.engines['portfolio']['url']}/portfolio/analyze",
                            json={
                                "portfolio_id": self.test_data['trading_scenario']['portfolio_id'],
                                "strategy_impact": strategy,
                                "current_positions": self.test_data['trading_scenario']['positions']
                            },
                            timeout=60
                        ) as response:
                            step3_time = (time.time() - step3_start) * 1000
                            
                            if response.status == 200:
                                data = await response.json()
                                step3_result = {
                                    "step": "portfolio_analysis",
                                    "success": True,
                                    "response_time_ms": step3_time,
                                    "analysis_metrics": len(data.get("metrics", {}))
                                }
                            else:
                                step3_result = {"step": "portfolio_analysis", "success": False, "error": f"HTTP {response.status}"}
                    
                    except Exception as e:
                        step3_result = {"step": "portfolio_analysis", "success": False, "error": str(e)}
                    
                    results["execution_steps"].append(step3_result)
                    
                    # Step 4: Execute strategy
                    if step3_result.get("success"):
                        step4_start = time.time()
                        try:
                            async with session.post(
                                f"{self.engines['strategy']['url']}/strategy/execute",
                                json={
                                    "strategy_id": strategy["strategy_id"],
                                    "market_data": self.test_data['live_market_data'][:20],
                                    "execution_mode": "simulation"
                                },
                                timeout=90
                            ) as response:
                                step4_time = (time.time() - step4_start) * 1000
                                
                                if response.status == 200:
                                    data = await response.json()
                                    step4_result = {
                                        "step": "strategy_execution",
                                        "success": True,
                                        "response_time_ms": step4_time,
                                        "trades_generated": data.get("trades_generated", 0)
                                    }
                                    results["strategy_performance"] = data.get("performance", {})
                                else:
                                    step4_result = {"step": "strategy_execution", "success": False, "error": f"HTTP {response.status}"}
                        
                        except Exception as e:
                            step4_result = {"step": "strategy_execution", "success": False, "error": str(e)}
                        
                        results["execution_steps"].append(step4_result)
            
            # Calculate integration success metrics
            successful_steps = [step for step in results["execution_steps"] if step.get("success")]
            results["integration_success"] = {
                "steps_completed": len(successful_steps),
                "total_steps": len(results["execution_steps"]),
                "success_rate": (len(successful_steps) / len(results["execution_steps"])) * 100 if results["execution_steps"] else 0,
                "total_execution_time": sum(step.get("response_time_ms", 0) for step in successful_steps)
            }
        
        return results
    
    async def test_factor_to_ml_pipeline(self) -> Dict[str, Any]:
        """Test factor calculation to ML prediction pipeline"""
        logger.info("Testing factor to ML pipeline integration...")
        
        results = {
            "test_name": "factor_to_ml_pipeline",
            "test_timestamp": datetime.now().isoformat(),
            "pipeline_steps": [],
            "ml_predictions": {}
        }
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Calculate factors
            step1_start = time.time()
            try:
                async with session.post(
                    f"{self.engines['factor']['url']}/factors/calculate/AAPL",
                    json={"factor_ids": ["RSI_14", "MACD_SIGNAL", "PE_RATIO"]},
                    timeout=60
                ) as response:
                    step1_time = (time.time() - step1_start) * 1000
                    
                    if response.status == 200:
                        step1_result = {
                            "step": "factor_calculation",
                            "success": True,
                            "response_time_ms": step1_time,
                            "symbol": "AAPL"
                        }
                    else:
                        step1_result = {"step": "factor_calculation", "success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                step1_result = {"step": "factor_calculation", "success": False, "error": str(e)}
            
            results["pipeline_steps"].append(step1_result)
            
            # Step 2: Extract features
            if step1_result.get("success"):
                await asyncio.sleep(1)  # Allow factor calculation to complete
                
                step2_start = time.time()
                try:
                    price_data = list(self.test_data['portfolio'].get('AAPL', {}).items())[-50:]
                    formatted_price_data = [
                        {
                            'timestamp': date,
                            'open': data['Open'],
                            'high': data['High'],
                            'low': data['Low'],
                            'close': data['Close'],
                            'volume': data['Volume']
                        }
                        for date, data in price_data
                    ]
                    
                    async with session.post(
                        f"{self.engines['features']['url']}/features/calculate/RSI",
                        json={
                            "symbol": "AAPL",
                            "indicator": "RSI",
                            "price_data": formatted_price_data,
                            "parameters": {"period": 14}
                        },
                        timeout=30
                    ) as response:
                        step2_time = (time.time() - step2_start) * 1000
                        
                        if response.status == 200:
                            step2_result = {
                                "step": "feature_extraction",
                                "success": True,
                                "response_time_ms": step2_time,
                                "indicator": "RSI"
                            }
                        else:
                            step2_result = {"step": "feature_extraction", "success": False, "error": f"HTTP {response.status}"}
                
                except Exception as e:
                    step2_result = {"step": "feature_extraction", "success": False, "error": str(e)}
                
                results["pipeline_steps"].append(step2_result)
                
                # Step 3: ML prediction
                if step2_result.get("success"):
                    step3_start = time.time()
                    try:
                        # Generate features for ML prediction
                        price_history = [data['Close'] for _, data in price_data]
                        returns = np.diff(price_history) / price_history[:-1] if len(price_history) > 1 else [0]
                        
                        ml_features = {
                            "current_price": price_history[-1] if price_history else 150,
                            "price_change_5d": (price_history[-1] - price_history[-6]) / price_history[-6] if len(price_history) >= 6 else 0,
                            "volatility_20d": np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns) if returns else 0.02,
                            "rsi_14": 50 + np.random.normal(0, 10),  # Mock RSI from features engine
                            "volume_ratio": 1.0 + np.random.normal(0, 0.2)
                        }
                        
                        async with session.post(
                            f"{self.engines['ml']['url']}/ml/predict/price",
                            json={
                                "symbol": "AAPL",
                                "features": ml_features,
                                "prediction_horizon": "1d"
                            },
                            timeout=30
                        ) as response:
                            step3_time = (time.time() - step3_start) * 1000
                            
                            if response.status == 200:
                                data = await response.json()
                                step3_result = {
                                    "step": "ml_prediction",
                                    "success": True,
                                    "response_time_ms": step3_time,
                                    "prediction_type": "price_prediction"
                                }
                                results["ml_predictions"] = {
                                    "predicted_direction": data.get("predicted_direction"),
                                    "confidence": data.get("confidence"),
                                    "expected_return": data.get("expected_return")
                                }
                            else:
                                step3_result = {"step": "ml_prediction", "success": False, "error": f"HTTP {response.status}"}
                    
                    except Exception as e:
                        step3_result = {"step": "ml_prediction", "success": False, "error": str(e)}
                    
                    results["pipeline_steps"].append(step3_result)
            
            # Calculate pipeline performance
            successful_steps = [step for step in results["pipeline_steps"] if step.get("success")]
            results["pipeline_performance"] = {
                "steps_completed": len(successful_steps),
                "total_steps": len(results["pipeline_steps"]),
                "success_rate": (len(successful_steps) / len(results["pipeline_steps"])) * 100 if results["pipeline_steps"] else 0,
                "total_pipeline_time": sum(step.get("response_time_ms", 0) for step in successful_steps)
            }
        
        return results
    
    async def test_websocket_real_time_integration(self) -> Dict[str, Any]:
        """Test real-time WebSocket integration with other engines"""
        logger.info("Testing WebSocket real-time integration...")
        
        results = {
            "test_name": "websocket_real_time_integration",
            "test_timestamp": datetime.now().isoformat(),
            "websocket_tests": [],
            "real_time_performance": {}
        }
        
        async with aiohttp.ClientSession() as session:
            # Test WebSocket subscription setup
            subscription_setup_start = time.time()
            try:
                async with session.post(
                    f"{self.engines['websocket']['url']}/websocket/subscribe",
                    json={
                        "client_id": "integration_test_client",
                        "subscriptions": ["market_data", "risk_alerts", "portfolio_updates"],
                        "symbols": ["AAPL", "MSFT"]
                    },
                    timeout=10
                ) as response:
                    setup_time = (time.time() - subscription_setup_start) * 1000
                    
                    if response.status == 200:
                        ws_setup_result = {
                            "test": "websocket_subscription_setup",
                            "success": True,
                            "response_time_ms": setup_time,
                            "subscriptions": 3
                        }
                    else:
                        ws_setup_result = {"test": "websocket_subscription_setup", "success": False, "error": f"HTTP {response.status}"}
            
            except Exception as e:
                ws_setup_result = {"test": "websocket_subscription_setup", "success": False, "error": str(e)}
            
            results["websocket_tests"].append(ws_setup_result)
            
            # Test market data broadcasting
            if ws_setup_result.get("success"):
                broadcast_start = time.time()
                try:
                    async with session.post(
                        f"{self.engines['websocket']['url']}/websocket/broadcast",
                        json={
                            "topic": "market_data",
                            "data": self.test_data['live_market_data'][:5],
                            "target_clients": ["integration_test_client"]
                        },
                        timeout=10
                    ) as response:
                        broadcast_time = (time.time() - broadcast_start) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            broadcast_result = {
                                "test": "market_data_broadcast",
                                "success": True,
                                "response_time_ms": broadcast_time,
                                "messages_broadcast": 5,
                                "clients_reached": data.get("clients_reached", 0)
                            }
                        else:
                            broadcast_result = {"test": "market_data_broadcast", "success": False, "error": f"HTTP {response.status}"}
                
                except Exception as e:
                    broadcast_result = {"test": "market_data_broadcast", "success": False, "error": str(e)}
                
                results["websocket_tests"].append(broadcast_result)
                
                # Test risk alert integration
                risk_alert_start = time.time()
                try:
                    # Trigger a risk check that might generate an alert
                    async with session.post(
                        f"{self.engines['risk']['url']}/risk/check/websocket_integration_test",
                        json={
                            "position_data": {
                                "AAPL": {"shares": 10000, "price": 150, "market_value": 1500000}  # Large position
                            },
                            "risk_limits": {"max_position_size": 100000}  # Lower limit to trigger alert
                        },
                        timeout=20
                    ) as response:
                        risk_check_time = (time.time() - risk_alert_start) * 1000
                        
                        if response.status == 200:
                            # Now test if WebSocket can broadcast risk alert
                            alert_data = {
                                "alert_type": "position_limit_breach",
                                "portfolio_id": "websocket_integration_test",
                                "severity": "HIGH",
                                "message": "Position size limit exceeded"
                            }
                            
                            async with session.post(
                                f"{self.engines['websocket']['url']}/websocket/broadcast",
                                json={
                                    "topic": "risk_alerts",
                                    "data": [alert_data],
                                    "target_clients": ["integration_test_client"]
                                },
                                timeout=10
                            ) as ws_response:
                                if ws_response.status == 200:
                                    risk_alert_result = {
                                        "test": "risk_alert_integration",
                                        "success": True,
                                        "response_time_ms": risk_check_time,
                                        "alert_broadcasted": True
                                    }
                                else:
                                    risk_alert_result = {"test": "risk_alert_integration", "success": False, "error": f"WS HTTP {ws_response.status}"}
                        else:
                            risk_alert_result = {"test": "risk_alert_integration", "success": False, "error": f"Risk HTTP {response.status}"}
                
                except Exception as e:
                    risk_alert_result = {"test": "risk_alert_integration", "success": False, "error": str(e)}
                
                results["websocket_tests"].append(risk_alert_result)
            
            # Calculate WebSocket integration performance
            successful_tests = [test for test in results["websocket_tests"] if test.get("success")]
            results["real_time_performance"] = {
                "tests_passed": len(successful_tests),
                "total_tests": len(results["websocket_tests"]),
                "success_rate": (len(successful_tests) / len(results["websocket_tests"])) * 100 if results["websocket_tests"] else 0,
                "avg_response_time": statistics.mean([test.get("response_time_ms", 0) for test in successful_tests]) if successful_tests else 0
            }
        
        return results
    
    async def test_end_to_end_integration_flow(self) -> Dict[str, Any]:
        """Test complete end-to-end integration across all engines"""
        logger.info("Testing end-to-end integration flow...")
        
        results = {
            "test_name": "end_to_end_integration_flow",
            "test_timestamp": datetime.now().isoformat(),
            "flow_execution": [],
            "system_performance": {}
        }
        
        overall_start_time = time.time()
        
        # Execute a complete trading workflow
        async with aiohttp.ClientSession() as session:
            # 1. Ingest market data
            logger.info("Step 1: Ingesting market data...")
            step1_start = time.time()
            market_data_result = await self.execute_market_data_ingestion(session)
            step1_time = (time.time() - step1_start) * 1000
            market_data_result["step"] = "market_data_ingestion"
            market_data_result["execution_time_ms"] = step1_time
            results["flow_execution"].append(market_data_result)
            
            if market_data_result.get("success"):
                await asyncio.sleep(1)  # Data propagation delay
                
                # 2. Calculate factors and features
                logger.info("Step 2: Calculating factors and features...")
                step2_start = time.time()
                factor_result = await self.execute_factor_calculation(session)
                step2_time = (time.time() - step2_start) * 1000
                factor_result["step"] = "factor_and_features_calculation"
                factor_result["execution_time_ms"] = step2_time
                results["flow_execution"].append(factor_result)
                
                if factor_result.get("success"):
                    await asyncio.sleep(1)
                    
                    # 3. ML predictions
                    logger.info("Step 3: Generating ML predictions...")
                    step3_start = time.time()
                    ml_result = await self.execute_ml_prediction(session)
                    step3_time = (time.time() - step3_start) * 1000
                    ml_result["step"] = "ml_prediction"
                    ml_result["execution_time_ms"] = step3_time
                    results["flow_execution"].append(ml_result)
                    
                    if ml_result.get("success"):
                        await asyncio.sleep(1)
                        
                        # 4. Strategy execution
                        logger.info("Step 4: Executing trading strategy...")
                        step4_start = time.time()
                        strategy_result = await self.execute_strategy(session)
                        step4_time = (time.time() - step4_start) * 1000
                        strategy_result["step"] = "strategy_execution"
                        strategy_result["execution_time_ms"] = step4_time
                        results["flow_execution"].append(strategy_result)
                        
                        if strategy_result.get("success"):
                            await asyncio.sleep(1)
                            
                            # 5. Risk validation
                            logger.info("Step 5: Risk validation...")
                            step5_start = time.time()
                            risk_result = await self.execute_risk_validation(session)
                            step5_time = (time.time() - step5_start) * 1000
                            risk_result["step"] = "risk_validation"
                            risk_result["execution_time_ms"] = step5_time
                            results["flow_execution"].append(risk_result)
                            
                            if risk_result.get("success"):
                                await asyncio.sleep(1)
                                
                                # 6. Portfolio optimization
                                logger.info("Step 6: Portfolio optimization...")
                                step6_start = time.time()
                                portfolio_result = await self.execute_portfolio_optimization(session)
                                step6_time = (time.time() - step6_start) * 1000
                                portfolio_result["step"] = "portfolio_optimization"
                                portfolio_result["execution_time_ms"] = step6_time
                                results["flow_execution"].append(portfolio_result)
                                
                                if portfolio_result.get("success"):
                                    await asyncio.sleep(0.5)
                                    
                                    # 7. Analytics and reporting
                                    logger.info("Step 7: Analytics and reporting...")
                                    step7_start = time.time()
                                    analytics_result = await self.execute_analytics_reporting(session)
                                    step7_time = (time.time() - step7_start) * 1000
                                    analytics_result["step"] = "analytics_reporting"
                                    analytics_result["execution_time_ms"] = step7_time
                                    results["flow_execution"].append(analytics_result)
                                    
                                    # 8. Real-time distribution
                                    logger.info("Step 8: Real-time distribution...")
                                    step8_start = time.time()
                                    websocket_result = await self.execute_realtime_distribution(session)
                                    step8_time = (time.time() - step8_start) * 1000
                                    websocket_result["step"] = "realtime_distribution"
                                    websocket_result["execution_time_ms"] = step8_time
                                    results["flow_execution"].append(websocket_result)
        
        # Calculate overall system performance
        total_execution_time = time.time() - overall_start_time
        successful_steps = [step for step in results["flow_execution"] if step.get("success")]
        
        results["system_performance"] = {
            "total_execution_time_sec": total_execution_time,
            "steps_completed": len(successful_steps),
            "total_steps": len(results["flow_execution"]),
            "end_to_end_success_rate": (len(successful_steps) / len(results["flow_execution"])) * 100 if results["flow_execution"] else 0,
            "avg_step_time_ms": statistics.mean([step.get("execution_time_ms", 0) for step in successful_steps]) if successful_steps else 0,
            "engines_involved": len(self.engines),
            "data_flow_latency_ms": sum(step.get("execution_time_ms", 0) for step in successful_steps)
        }
        
        logger.info(f"End-to-end integration flow completed: {results['system_performance']['end_to_end_success_rate']:.1f}% success rate")
        
        return results
    
    # Helper methods for end-to-end flow execution
    async def execute_market_data_ingestion(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['marketdata']['url']}/marketdata/ingest",
                json={
                    "source": "e2e_test",
                    "data_type": "tick_data",
                    "market_data": self.test_data['live_market_data'][:20]
                },
                timeout=30
            ) as response:
                return {"success": response.status == 200, "data_points": 20}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_factor_calculation(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['factor']['url']}/factors/calculate/AAPL",
                json={"factor_ids": ["RSI_14", "MACD_SIGNAL"]},
                timeout=60
            ) as response:
                return {"success": response.status == 200, "factors_calculated": 2}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_ml_prediction(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['ml']['url']}/ml/predict/price",
                json={
                    "symbol": "AAPL",
                    "features": {
                        "current_price": 150,
                        "rsi_14": 65,
                        "volatility_20d": 0.25
                    }
                },
                timeout=30
            ) as response:
                return {"success": response.status == 200, "predictions_made": 1}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_strategy(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['strategy']['url']}/strategy/execute",
                json={
                    "strategy_id": self.test_data['test_strategy']['strategy_id'],
                    "market_data": self.test_data['live_market_data'][:10],
                    "execution_mode": "simulation"
                },
                timeout=60
            ) as response:
                return {"success": response.status == 200, "strategy_executed": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_risk_validation(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['risk']['url']}/risk/check/e2e_test_portfolio",
                json={
                    "position_data": self.test_data['trading_scenario']['positions'],
                    "risk_limits": self.test_data['trading_scenario']['risk_limits']
                },
                timeout=30
            ) as response:
                return {"success": response.status == 200, "risk_checks_performed": 1}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_portfolio_optimization(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['portfolio']['url']}/portfolio/optimize",
                json={
                    "assets": ["AAPL", "MSFT"],
                    "weights": [0.5, 0.5],
                    "optimization_method": "mean_variance",
                    "constraints": {"min_weight": 0.1, "max_weight": 0.9}
                },
                timeout=120
            ) as response:
                return {"success": response.status == 200, "portfolios_optimized": 1}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_analytics_reporting(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['analytics']['url']}/analytics/performance/e2e_test_portfolio",
                json={
                    "positions": self.test_data['trading_scenario']['positions'],
                    "benchmark": "SPY"
                },
                timeout=30
            ) as response:
                return {"success": response.status == 200, "analytics_generated": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_realtime_distribution(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            async with session.post(
                f"{self.engines['websocket']['url']}/websocket/broadcast",
                json={
                    "topic": "portfolio_updates",
                    "data": [{"portfolio_id": "e2e_test_portfolio", "update": "optimization_complete"}],
                    "target_clients": ["all"]
                },
                timeout=10
            ) as response:
                return {"success": response.status == 200, "messages_distributed": 1}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all inter-engine integration tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE INTER-ENGINE INTEGRATION TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "Inter-Engine Integration Comprehensive Test",
            "test_start_time": datetime.now().isoformat(),
            "engines_tested": list(self.engines.keys()),
            "integration_flows_tested": [flow["name"] for flow in self.integration_flows],
            "test_results": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Test sequence
        test_sequence = [
            ("all_engines_health_check", self.check_all_engines_health),
            ("market_data_flow_integration", self.test_market_data_flow_integration),
            ("strategy_execution_integration", self.test_strategy_execution_integration),
            ("factor_to_ml_pipeline", self.test_factor_to_ml_pipeline),
            ("websocket_real_time_integration", self.test_websocket_real_time_integration),
            ("end_to_end_integration_flow", self.test_end_to_end_integration_flow)
        ]
        
        for test_name, test_function in test_sequence:
            logger.info(f"Running integration test: {test_name}")
            try:
                test_result = await test_function()
                comprehensive_results["test_results"][test_name] = test_result
                logger.info(f"Completed integration test: {test_name} ✅")
            except Exception as e:
                logger.error(f"Failed integration test: {test_name} ❌ - {e}")
                comprehensive_results["test_results"][test_name] = {
                    "error": str(e),
                    "test_failed": True
                }
            
            # Brief pause between tests
            await asyncio.sleep(3)
        
        # Calculate performance summary
        total_test_time = time.time() - start_time
        comprehensive_results["total_test_time_seconds"] = total_test_time
        comprehensive_results["test_end_time"] = datetime.now().isoformat()
        
        # Analyze results and create performance summary
        self._create_integration_summary(comprehensive_results)
        
        logger.info("="*80)
        logger.info("INTER-ENGINE INTEGRATION TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_integration_summary(self, results: Dict[str, Any]):
        """Create an integration performance summary"""
        summary = {
            "overall_integration_health": "unknown",
            "engine_health_status": {},
            "integration_flow_performance": {},
            "system_coordination": {},
            "end_to_end_metrics": {}
        }
        
        # Engine health analysis
        health_test = results["test_results"].get("all_engines_health_check", {})
        if not health_test.get("test_failed"):
            summary["engine_health_status"] = {
                "healthy_engines": health_test.get("healthy_engines", 0),
                "total_engines": health_test.get("total_engines", 0),
                "health_percentage": health_test.get("health_percentage", 0)
            }
        
        # Integration flow analysis
        integration_tests = [
            ("market_data_flow", "market_data_flow_integration"),
            ("strategy_execution", "strategy_execution_integration"),
            ("factor_ml_pipeline", "factor_to_ml_pipeline"),
            ("websocket_realtime", "websocket_real_time_integration")
        ]
        
        for flow_name, test_name in integration_tests:
            test_result = results["test_results"].get(test_name, {})
            if not test_result.get("test_failed"):
                if "integration_success" in test_result:
                    summary["integration_flow_performance"][flow_name] = test_result["integration_success"]
                elif "pipeline_performance" in test_result:
                    summary["integration_flow_performance"][flow_name] = test_result["pipeline_performance"]
                elif "real_time_performance" in test_result:
                    summary["integration_flow_performance"][flow_name] = test_result["real_time_performance"]
        
        # End-to-end analysis
        e2e_test = results["test_results"].get("end_to_end_integration_flow", {})
        if not e2e_test.get("test_failed"):
            summary["end_to_end_metrics"] = e2e_test.get("system_performance", {})
        
        # Overall integration health assessment
        health_percentage = summary["engine_health_status"].get("health_percentage", 0)
        flow_success_rates = [
            flow.get("success_rate", 0) 
            for flow in summary["integration_flow_performance"].values()
        ]
        
        e2e_success_rate = summary["end_to_end_metrics"].get("end_to_end_success_rate", 0)
        
        if health_percentage >= 90 and e2e_success_rate >= 80 and all(rate >= 80 for rate in flow_success_rates):
            summary["overall_integration_health"] = "excellent"
        elif health_percentage >= 80 and e2e_success_rate >= 70 and all(rate >= 70 for rate in flow_success_rates):
            summary["overall_integration_health"] = "good"
        elif health_percentage >= 70 and e2e_success_rate >= 50:
            summary["overall_integration_health"] = "fair"
        else:
            summary["overall_integration_health"] = "poor"
        
        results["performance_summary"] = summary

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/inter_engine_integration_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = InterEngineIntegrationTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("INTER-ENGINE INTEGRATION TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Integration Health: {summary.get('overall_integration_health', 'unknown').upper()}")
    
    engine_health = summary.get("engine_health_status", {})
    if engine_health:
        print(f"\nEngine Health Status:")
        print(f"  Healthy Engines: {engine_health.get('healthy_engines', 0)}/{engine_health.get('total_engines', 0)}")
        print(f"  Health Percentage: {engine_health.get('health_percentage', 0):.1f}%")
    
    flow_perf = summary.get("integration_flow_performance", {})
    if flow_perf:
        print(f"\nIntegration Flow Performance:")
        for flow_name, metrics in flow_perf.items():
            print(f"  {flow_name.replace('_', ' ').title()}:")
            print(f"    Success Rate: {metrics.get('success_rate', 0):.1f}%")
            print(f"    Steps Completed: {metrics.get('steps_completed', 0)}/{metrics.get('total_steps', 0)}")
    
    e2e_metrics = summary.get("end_to_end_metrics", {})
    if e2e_metrics:
        print(f"\nEnd-to-End Metrics:")
        print(f"  Total Execution Time: {e2e_metrics.get('total_execution_time_sec', 0):.2f} seconds")
        print(f"  Success Rate: {e2e_metrics.get('end_to_end_success_rate', 0):.1f}%")
        print(f"  Steps Completed: {e2e_metrics.get('steps_completed', 0)}/{e2e_metrics.get('total_steps', 0)}")
        print(f"  Engines Involved: {e2e_metrics.get('engines_involved', 0)}")
        print(f"  Data Flow Latency: {e2e_metrics.get('data_flow_latency_ms', 0):.1f}ms")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
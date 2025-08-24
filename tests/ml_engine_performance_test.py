#!/usr/bin/env python3
"""
ML Engine Comprehensive Performance Test
Tests the containerized ML Engine performance, capabilities, and integration
Focuses on ML inference, model predictions, and Neural Engine optimization
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

class MLEnginePerformanceTest:
    """
    Comprehensive performance testing suite for ML Engine
    Tests containerized ML inference, prediction capabilities, and model performance
    """
    
    def __init__(self):
        self.base_url = "http://localhost:8400"
        self.docker_client = docker.from_env()
        self.container_name = "nautilus-ml-engine"
        self.test_data = {}
        self.performance_metrics = {}
        self.load_test_data()
        
    def load_test_data(self):
        """Load comprehensive test data for ML engine testing"""
        logger.info("Loading test data for ML Engine performance testing...")
        
        try:
            # Load portfolio data for ML predictions
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
            
            # Generate ML-specific test data
            self.generate_ml_test_data()
            
            # Load high-frequency data for real-time predictions
            try:
                with open('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/data/synthetic_AAPL_hf.json', 'r') as f:
                    hf_data = json.load(f)
                    self.test_data['high_frequency'] = hf_data[:500]  # First 500 records
            except FileNotFoundError:
                self.generate_synthetic_hf_data()
            
            logger.info(f"Loaded test data for {len(portfolio_data)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.generate_synthetic_test_data()
    
    def generate_ml_test_data(self):
        """Generate ML-specific test data for various prediction types"""
        np.random.seed(42)
        
        # Generate price prediction features
        self.test_data['price_features'] = {}
        symbols = list(self.test_data['portfolio'].keys())[:5]
        
        for symbol in symbols:
            price_data = self.test_data['portfolio'][symbol]
            if not price_data:
                continue
                
            # Extract price series
            dates = sorted(price_data.keys())[-100:]  # Last 100 days
            closes = [price_data[date]['Close'] for date in dates]
            highs = [price_data[date]['High'] for date in dates]
            lows = [price_data[date]['Low'] for date in dates]
            volumes = [price_data[date]['Volume'] for date in dates]
            
            # Calculate technical features
            returns = np.diff(closes) / closes[:-1]
            rsi = self.calculate_rsi(closes, 14)
            sma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            
            features = {
                'current_price': closes[-1],
                'price_change_5d': (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0,
                'price_change_20d': (closes[-1] - closes[-21]) / closes[-21] if len(closes) >= 21 else 0,
                'rsi_14': rsi[-1] if len(rsi) > 0 else 50,
                'sma_20': sma_20[-1] if len(sma_20) > 0 else closes[-1],
                'volatility_20d': volatility,
                'volume_ratio': volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1,
                'high_low_ratio': (highs[-1] - lows[-1]) / closes[-1],
                'returns_1d': returns[-1] if len(returns) > 0 else 0,
                'returns_5d': np.mean(returns[-5:]) if len(returns) >= 5 else 0
            }
            
            self.test_data['price_features'][symbol] = features
        
        # Generate market regime features
        self.test_data['regime_features'] = {
            'vix_level': np.random.uniform(15, 40),
            'yield_curve_slope': np.random.normal(1.5, 0.5),
            'credit_spreads': np.random.uniform(100, 500),
            'market_momentum': np.random.normal(0, 0.2),
            'sector_rotation': np.random.uniform(-1, 1),
            'fear_greed_index': np.random.uniform(20, 80)
        }
        
        # Generate volatility forecasting data
        self.test_data['volatility_data'] = {}
        for symbol in symbols[:3]:
            price_data = self.test_data['portfolio'][symbol]
            if price_data:
                dates = sorted(price_data.keys())[-50:]
                closes = [price_data[date]['Close'] for date in dates]
                returns = np.diff(closes) / closes[:-1]
                
                self.test_data['volatility_data'][symbol] = {
                    'historical_returns': returns.tolist(),
                    'realized_volatility': np.std(returns) * np.sqrt(252),  # Annualized
                    'garch_params': {
                        'omega': 0.00001,
                        'alpha': 0.05,
                        'beta': 0.90
                    }
                }
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return []
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / np.where(avg_losses == 0, 1, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.tolist()
    
    def generate_synthetic_test_data(self):
        """Generate synthetic test data if real data is not available"""
        logger.info("Generating synthetic test data for ML testing...")
        
        np.random.seed(42)
        
        # Generate portfolio data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        portfolio_data = {}
        
        for symbol in symbols:
            # Generate 1 year of daily data
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            
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
        self.generate_ml_test_data()
        self.generate_synthetic_hf_data()
    
    def generate_synthetic_hf_data(self):
        """Generate synthetic high-frequency data for real-time predictions"""
        np.random.seed(42)
        hf_data = []
        
        base_time = datetime.now() - timedelta(hours=1)
        for i in range(500):
            timestamp = base_time + timedelta(seconds=i * 7.2)  # ~7 second intervals
            price = 150.0 + np.random.normal(0, 0.5)
            
            hf_data.append({
                'timestamp': timestamp.isoformat(),
                'symbol': 'AAPL',
                'price': round(price, 2),
                'volume': int(np.random.exponential(10000)),
                'bid': round(price - 0.01, 2),
                'ask': round(price + 0.01, 2)
            })
        
        self.test_data['high_frequency'] = hf_data
    
    async def check_container_health(self) -> Dict[str, Any]:
        """Check if the ML Engine container is healthy"""
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
        """Test the health endpoint and ML engine status"""
        logger.info("Testing ML Engine health endpoint...")
        
        results = {
            "test_name": "health_endpoint",
            "test_timestamp": datetime.now().isoformat(),
            "response_times": [],
            "success_rate": 0.0,
            "models_loaded": 0,
            "predictions_made": 0,
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
                                results["models_loaded"] = data.get("models_loaded", 0)
                                results["predictions_made"] = data.get("predictions_made", 0)
                                results["messagebus_connected"] = data.get("messagebus_connected", False)
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
    
    async def test_price_prediction_performance(self) -> Dict[str, Any]:
        """Test price prediction performance with various input sizes"""
        logger.info("Testing price prediction performance...")
        
        results = {
            "test_name": "price_prediction_performance",
            "test_timestamp": datetime.now().isoformat(),
            "prediction_tests": [],
            "throughput_predictions_per_sec": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            # Test price predictions for different symbols
            symbols = list(self.test_data.get('price_features', {}).keys())
            
            for symbol in symbols[:5]:  # Test first 5 symbols
                features = self.test_data['price_features'][symbol]
                
                prediction_data = {
                    "symbol": symbol,
                    "features": features,
                    "prediction_horizon": "1d",
                    "model_type": "gradient_boosting"
                }
                
                # Test multiple predictions for this symbol
                response_times = []
                successful_predictions = 0
                
                for i in range(3):  # 3 predictions per symbol
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.base_url}/ml/predict/price",
                            json=prediction_data,
                            timeout=30
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            response_times.append(response_time)
                            
                            if response.status == 200:
                                successful_predictions += 1
                                data = await response.json()
                                logger.info(f"Price prediction {symbol} #{i+1}: {response_time:.2f}ms - "
                                          f"Direction: {data.get('predicted_direction', 'N/A')}")
                            else:
                                logger.warning(f"Price prediction {symbol} #{i+1}: HTTP {response.status}")
                    
                    except Exception as e:
                        logger.error(f"Price prediction {symbol} #{i+1} failed: {e}")
                    
                    await asyncio.sleep(0.1)
                
                # Calculate statistics for this symbol
                symbol_result = {
                    "symbol": symbol,
                    "successful_predictions": successful_predictions,
                    "total_attempts": 3,
                    "success_rate": (successful_predictions / 3) * 100,
                    "avg_response_time": statistics.mean(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0
                }
                
                results["prediction_tests"].append(symbol_result)
        
        # Calculate overall throughput
        total_successful = sum(r["successful_predictions"] for r in results["prediction_tests"])
        total_time = sum(r["avg_response_time"] / 1000 for r in results["prediction_tests"])
        if total_time > 0:
            results["throughput_predictions_per_sec"] = total_successful / total_time
        
        return results
    
    async def test_market_regime_detection(self) -> Dict[str, Any]:
        """Test market regime detection capabilities"""
        logger.info("Testing market regime detection...")
        
        results = {
            "test_name": "market_regime_detection",
            "test_timestamp": datetime.now().isoformat(),
            "regime_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test different market scenarios
            market_scenarios = [
                {
                    "name": "high_volatility_market",
                    "features": {
                        **self.test_data['regime_features'],
                        'vix_level': 35,
                        'market_momentum': -0.15,
                        'fear_greed_index': 25
                    }
                },
                {
                    "name": "low_volatility_market",
                    "features": {
                        **self.test_data['regime_features'],
                        'vix_level': 15,
                        'market_momentum': 0.05,
                        'fear_greed_index': 70
                    }
                },
                {
                    "name": "crisis_market",
                    "features": {
                        **self.test_data['regime_features'],
                        'vix_level': 45,
                        'credit_spreads': 400,
                        'market_momentum': -0.25,
                        'fear_greed_index': 10
                    }
                }
            ]
            
            for scenario in market_scenarios:
                logger.info(f"Testing regime detection: {scenario['name']}")
                
                regime_data = {
                    "market_features": scenario["features"],
                    "lookback_period": 252,
                    "model_type": "hmm_regime"
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/ml/predict/regime",
                        json=regime_data,
                        timeout=30
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            regime_result = {
                                "scenario_name": scenario["name"],
                                "success": True,
                                "response_time_ms": response_time,
                                "detected_regime": data.get("current_regime", "UNKNOWN"),
                                "confidence": data.get("confidence", 0),
                                "regime_probability": data.get("regime_probability", {})
                            }
                            logger.info(f"Regime detection {scenario['name']}: {response_time:.2f}ms - "
                                      f"Regime: {regime_result['detected_regime']}")
                        else:
                            regime_result = {
                                "scenario_name": scenario["name"],
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    regime_result = {
                        "scenario_name": scenario["name"],
                        "success": False,
                        "error": str(e)
                    }
                
                results["regime_tests"].append(regime_result)
                await asyncio.sleep(0.2)
        
        return results
    
    async def test_volatility_forecasting(self) -> Dict[str, Any]:
        """Test volatility forecasting capabilities"""
        logger.info("Testing volatility forecasting...")
        
        results = {
            "test_name": "volatility_forecasting",
            "test_timestamp": datetime.now().isoformat(),
            "volatility_tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test volatility forecasting for different symbols
            volatility_data = self.test_data.get('volatility_data', {})
            
            for symbol, vol_info in volatility_data.items():
                logger.info(f"Testing volatility forecasting for {symbol}")
                
                forecast_data = {
                    "symbol": symbol,
                    "historical_returns": vol_info["historical_returns"],
                    "forecast_horizon": 5,  # 5 days
                    "model_type": "garch",
                    "confidence_interval": 0.95
                }
                
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/ml/predict/volatility",
                        json=forecast_data,
                        timeout=60
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            vol_result = {
                                "symbol": symbol,
                                "success": True,
                                "response_time_ms": response_time,
                                "forecast_volatility": data.get("forecast_volatility", 0),
                                "confidence_bounds": data.get("confidence_bounds", {}),
                                "model_fit_quality": data.get("model_fit_quality", 0)
                            }
                            logger.info(f"Volatility forecast {symbol}: {response_time:.2f}ms - "
                                      f"Vol: {vol_result['forecast_volatility']:.4f}")
                        else:
                            vol_result = {
                                "symbol": symbol,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }
                
                except Exception as e:
                    vol_result = {
                        "symbol": symbol,
                        "success": False,
                        "error": str(e)
                    }
                
                results["volatility_tests"].append(vol_result)
                await asyncio.sleep(0.3)
        
        return results
    
    async def test_concurrent_ml_predictions(self, concurrent_users: int = 5, predictions_per_user: int = 4) -> Dict[str, Any]:
        """Test concurrent ML prediction capabilities"""
        logger.info(f"Testing concurrent ML predictions: {concurrent_users} users, {predictions_per_user} predictions each")
        
        results = {
            "test_name": "concurrent_ml_predictions",
            "test_timestamp": datetime.now().isoformat(),
            "concurrent_users": concurrent_users,
            "predictions_per_user": predictions_per_user,
            "total_predictions": concurrent_users * predictions_per_user,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "response_times": [],
            "prediction_breakdown": {}
        }
        
        async def user_predictions(user_id: int):
            """Simulate a user making multiple ML predictions"""
            user_results = {"successful": 0, "failed": 0, "response_times": [], "predictions": []}
            
            async with aiohttp.ClientSession() as session:
                for pred_id in range(predictions_per_user):
                    # Rotate between different prediction types
                    if pred_id % 3 == 0:
                        # Price prediction
                        prediction_type = "price_prediction"
                        symbol = list(self.test_data.get('price_features', {}).keys())[pred_id % 3] if self.test_data.get('price_features') else 'AAPL'
                        features = self.test_data.get('price_features', {}).get(symbol, {
                            'current_price': 150,
                            'rsi_14': 50,
                            'volatility_20d': 0.25
                        })
                        
                        start_time = time.time()
                        try:
                            async with session.post(
                                f"{self.base_url}/ml/predict/price",
                                json={"symbol": symbol, "features": features},
                                timeout=30
                            ) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["predictions"].append({"type": prediction_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["predictions"].append({"type": prediction_type, "success": False})
                    
                    elif pred_id % 3 == 1:
                        # Market regime detection
                        prediction_type = "regime_detection"
                        regime_features = self.test_data.get('regime_features', {
                            'vix_level': 25,
                            'market_momentum': 0.05
                        })
                        
                        start_time = time.time()
                        try:
                            async with session.post(
                                f"{self.base_url}/ml/predict/regime",
                                json={"market_features": regime_features},
                                timeout=30
                            ) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["predictions"].append({"type": prediction_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["predictions"].append({"type": prediction_type, "success": False})
                    
                    else:
                        # Volatility forecasting
                        prediction_type = "volatility_forecasting"
                        vol_data = list(self.test_data.get('volatility_data', {}).values())[0] if self.test_data.get('volatility_data') else {
                            "historical_returns": np.random.normal(0, 0.02, 30).tolist()
                        }
                        
                        start_time = time.time()
                        try:
                            async with session.post(
                                f"{self.base_url}/ml/predict/volatility",
                                json={
                                    "symbol": "AAPL",
                                    "historical_returns": vol_data.get("historical_returns", [])[:20],
                                    "forecast_horizon": 5
                                },
                                timeout=60
                            ) as response:
                                response_time = (time.time() - start_time) * 1000
                                user_results["response_times"].append(response_time)
                                if response.status == 200:
                                    user_results["successful"] += 1
                                else:
                                    user_results["failed"] += 1
                                user_results["predictions"].append({"type": prediction_type, "success": response.status == 200})
                        except:
                            user_results["failed"] += 1
                            user_results["predictions"].append({"type": prediction_type, "success": False})
                    
                    await asyncio.sleep(0.3)  # Brief delay between predictions
            
            return user_results
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [user_predictions(user_id) for user_id in range(concurrent_users)]
        user_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        total_test_time = time.time() - start_time
        
        # Aggregate results
        prediction_counts = {}
        for user_result in user_results_list:
            if isinstance(user_result, dict):
                results["successful_predictions"] += user_result["successful"]
                results["failed_predictions"] += user_result["failed"]
                results["response_times"].extend(user_result["response_times"])
                
                # Count prediction types
                for pred in user_result["predictions"]:
                    pred_type = pred["type"]
                    if pred_type not in prediction_counts:
                        prediction_counts[pred_type] = {"successful": 0, "failed": 0}
                    
                    if pred["success"]:
                        prediction_counts[pred_type]["successful"] += 1
                    else:
                        prediction_counts[pred_type]["failed"] += 1
        
        results["prediction_breakdown"] = prediction_counts
        
        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["p95_response_time"] = np.percentile(results["response_times"], 95)
        
        results["success_rate"] = (results["successful_predictions"] / results["total_predictions"]) * 100
        results["predictions_per_second"] = results["total_predictions"] / total_test_time
        results["total_test_time_seconds"] = total_test_time
        
        logger.info(f"Concurrent ML predictions test completed: {results['success_rate']:.1f}% success rate, "
                   f"{results['predictions_per_second']:.1f} predictions/sec")
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all ML engine performance tests"""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE ML ENGINE PERFORMANCE TEST")
        logger.info("="*80)
        
        start_time = time.time()
        
        comprehensive_results = {
            "test_suite_name": "ML Engine Comprehensive Performance Test",
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
            ("price_prediction_performance", self.test_price_prediction_performance),
            ("market_regime_detection", self.test_market_regime_detection),
            ("volatility_forecasting", self.test_volatility_forecasting),
            ("concurrent_ml_predictions", lambda: self.test_concurrent_ml_predictions(concurrent_users=3, predictions_per_user=4))
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
        logger.info("ML ENGINE PERFORMANCE TEST COMPLETE")
        logger.info(f"Total test time: {total_test_time:.2f} seconds")
        logger.info("="*80)
        
        return comprehensive_results
    
    def _create_performance_summary(self, results: Dict[str, Any]):
        """Create a performance summary from all test results"""
        summary = {
            "overall_health": "unknown",
            "ml_prediction_performance": {},
            "model_capabilities": {},
            "system_performance": {}
        }
        
        # Health endpoint analysis
        health_test = results["test_results"].get("health_endpoint", {})
        if not health_test.get("test_failed"):
            summary["system_performance"]["models_loaded"] = health_test.get("models_loaded", 0)
            summary["system_performance"]["predictions_made"] = health_test.get("predictions_made", 0)
            summary["system_performance"]["messagebus_connected"] = health_test.get("messagebus_connected", False)
        
        # Price prediction performance
        price_test = results["test_results"].get("price_prediction_performance", {})
        if not price_test.get("test_failed"):
            summary["ml_prediction_performance"]["price_throughput_per_sec"] = price_test.get("throughput_predictions_per_sec", 0)
            
            prediction_tests = price_test.get("prediction_tests", [])
            if prediction_tests:
                avg_success_rate = statistics.mean([t.get("success_rate", 0) for t in prediction_tests])
                avg_response_time = statistics.mean([t.get("avg_response_time", 0) for t in prediction_tests if t.get("avg_response_time", 0) > 0])
                summary["ml_prediction_performance"]["price_prediction_success_rate"] = avg_success_rate
                summary["ml_prediction_performance"]["price_prediction_avg_time_ms"] = avg_response_time
        
        # Model capabilities assessment
        regime_test = results["test_results"].get("market_regime_detection", {})
        if not regime_test.get("test_failed"):
            regime_tests = regime_test.get("regime_tests", [])
            successful_regime_tests = [t for t in regime_tests if t.get("success")]
            summary["model_capabilities"]["regime_detection_success_rate"] = (len(successful_regime_tests) / len(regime_tests)) * 100 if regime_tests else 0
        
        volatility_test = results["test_results"].get("volatility_forecasting", {})
        if not volatility_test.get("test_failed"):
            vol_tests = volatility_test.get("volatility_tests", [])
            successful_vol_tests = [t for t in vol_tests if t.get("success")]
            summary["model_capabilities"]["volatility_forecasting_success_rate"] = (len(successful_vol_tests) / len(vol_tests)) * 100 if vol_tests else 0
        
        # Concurrent performance analysis
        concurrent_test = results["test_results"].get("concurrent_ml_predictions", {})
        if not concurrent_test.get("test_failed"):
            summary["system_performance"]["concurrent_success_rate"] = concurrent_test.get("success_rate", 0)
            summary["system_performance"]["concurrent_predictions_per_sec"] = concurrent_test.get("predictions_per_second", 0)
        
        # Overall health assessment
        success_indicators = []
        
        if summary["ml_prediction_performance"].get("price_prediction_success_rate", 0) >= 80:
            success_indicators.append(1)
        if summary["model_capabilities"].get("regime_detection_success_rate", 0) >= 70:
            success_indicators.append(1)
        if summary["system_performance"].get("concurrent_success_rate", 0) >= 80:
            success_indicators.append(1)
        if summary["system_performance"].get("concurrent_predictions_per_sec", 0) > 1:
            success_indicators.append(1)
        
        if len(success_indicators) >= 3:
            summary["overall_health"] = "excellent"
        elif len(success_indicators) >= 2:
            summary["overall_health"] = "good"
        elif len(success_indicators) >= 1:
            summary["overall_health"] = "fair"
        else:
            summary["overall_health"] = "poor"
        
        results["performance_summary"] = summary

    def save_results_to_file(self, results: Dict[str, Any]):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/ml_engine_performance_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")

async def main():
    """Main execution function"""
    tester = MLEnginePerformanceTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    tester.save_results_to_file(results)
    
    # Print summary
    print("\n" + "="*80)
    print("ML ENGINE PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = results.get("performance_summary", {})
    print(f"Overall Health: {summary.get('overall_health', 'unknown').upper()}")
    
    system_perf = summary.get("system_performance", {})
    if system_perf:
        print(f"\nSystem Performance:")
        print(f"  Models Loaded: {system_perf.get('models_loaded', 0)}")
        print(f"  Predictions Made: {system_perf.get('predictions_made', 0)}")
        print(f"  MessageBus Connected: {system_perf.get('messagebus_connected', False)}")
        print(f"  Concurrent Success Rate: {system_perf.get('concurrent_success_rate', 0):.1f}%")
        print(f"  Concurrent Predictions/sec: {system_perf.get('concurrent_predictions_per_sec', 0):.1f}")
    
    ml_perf = summary.get("ml_prediction_performance", {})
    if ml_perf:
        print(f"\nML Prediction Performance:")
        print(f"  Price Prediction Success Rate: {ml_perf.get('price_prediction_success_rate', 0):.1f}%")
        print(f"  Price Prediction Avg Time: {ml_perf.get('price_prediction_avg_time_ms', 0):.1f}ms")
        print(f"  Price Throughput: {ml_perf.get('price_throughput_per_sec', 0):.1f} predictions/sec")
    
    capabilities = summary.get("model_capabilities", {})
    if capabilities:
        print(f"\nModel Capabilities:")
        print(f"  Regime Detection Success: {capabilities.get('regime_detection_success_rate', 0):.1f}%")
        print(f"  Volatility Forecasting Success: {capabilities.get('volatility_forecasting_success_rate', 0):.1f}%")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
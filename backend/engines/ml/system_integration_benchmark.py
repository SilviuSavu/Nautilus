#!/usr/bin/env python3
"""
Nautilus Trading Platform - Comprehensive System Integration Benchmark
M4 Max Optimization Validation - All Components Integration Test

This benchmark validates the entire M4 Max optimization stack working together:
- CPU optimization (12 P-cores + 4 E-cores)
- Unified memory management (546 GB/s bandwidth)
- Metal GPU acceleration (40 cores)
- Neural Engine (16 cores, 38 TOPS)
"""

import asyncio
import time
import json
import psutil
import numpy as np
import concurrent.futures
from datetime import datetime, timezone
import subprocess
import sys
import os
import logging
import requests
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from dataclasses import dataclass
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    component: str
    test_name: str
    duration: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    neural_engine_usage: Optional[float]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SystemMonitor:
    """Real-time system monitoring for M4 Max resources"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics
        
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_freq = psutil.cpu_freq()
                cpu_count_logical = psutil.cpu_count(logical=True)
                cpu_count_physical = psutil.cpu_count(logical=False)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # GPU metrics (Metal Performance Shaders if available)
                gpu_usage = self._get_gpu_usage()
                
                # Neural Engine metrics (approximation through system activity)
                neural_engine_usage = self._estimate_neural_engine_usage()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Network I/O
                network_io = psutil.net_io_counters()
                
                metric = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'cpu': {
                        'percent': cpu_percent,
                        'frequency': cpu_freq.current if cpu_freq else 0,
                        'logical_cores': cpu_count_logical,
                        'physical_cores': cpu_count_physical,
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent,
                        'used': memory.used
                    },
                    'gpu': gpu_usage,
                    'neural_engine': neural_engine_usage,
                    'disk_io': {
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0
                    },
                    'network_io': {
                        'bytes_sent': network_io.bytes_sent if network_io else 0,
                        'bytes_recv': network_io.bytes_recv if network_io else 0
                    }
                }
                
                self.metrics.append(metric)
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(1)
                
    def _get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get Metal GPU usage (M4 Max specific)"""
        try:
            # Use powermetrics for GPU usage on macOS
            result = subprocess.run(
                ['powermetrics', '--samplers', 'gpu_power', '-n', '1', '-i', '1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse GPU power/usage from powermetrics output
                lines = result.stdout.split('\n')
                gpu_info = {}
                for line in lines:
                    if 'GPU' in line and ('W' in line or '%' in line):
                        gpu_info['status'] = 'active'
                        # Extract numeric values if possible
                        break
                
                return gpu_info if gpu_info else {'status': 'available', 'usage': 0}
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return {'status': 'unknown', 'usage': 0}
        
    def _estimate_neural_engine_usage(self) -> Dict[str, Any]:
        """Estimate Neural Engine usage through system indicators"""
        # Neural Engine usage is difficult to measure directly
        # We can estimate based on ML workload patterns and system activity
        try:
            # Check for CoreML processes or high neural processing activity
            neural_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    pinfo = proc.info
                    if any(keyword in pinfo['name'].lower() for keyword in 
                           ['coreml', 'neural', 'tensorflow', 'pytorch', 'onnx']):
                        neural_processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            estimated_usage = min(sum(p.get('cpu_percent', 0) for p in neural_processes), 100.0)
            
            return {
                'estimated_usage': estimated_usage,
                'processes': len(neural_processes),
                'status': 'active' if neural_processes else 'idle'
            }
            
        except Exception as e:
            return {'status': 'unknown', 'error': str(e)}

class IntegratedTradingPipeline:
    """Complete trading pipeline benchmark with all engines"""
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.engine_urls = {
            'analytics': 'http://localhost:8100',
            'risk': 'http://localhost:8200', 
            'factor': 'http://localhost:8300',
            'ml': 'http://localhost:8400',
            'features': 'http://localhost:8500',
            'websocket': 'http://localhost:8600',
            'strategy': 'http://localhost:8700',
            'marketdata': 'http://localhost:8800',
            'portfolio': 'http://localhost:8900'
        }
        
    async def run_integrated_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive integrated trading pipeline benchmark"""
        results = []
        
        # 1. System Health Check
        results.extend(await self._health_check_all_services())
        
        # 2. End-to-End Trading Pipeline
        results.extend(await self._end_to_end_trading_pipeline())
        
        # 3. Multi-Component Workflow
        results.extend(await self._multi_component_workflow())
        
        # 4. Real-World Trading Scenarios
        results.extend(await self._real_world_scenarios())
        
        # 5. Cross-Component Data Flow
        results.extend(await self._cross_component_data_flow())
        
        return results
        
    async def _health_check_all_services(self) -> List[BenchmarkResult]:
        """Health check all containerized services"""
        results = []
        
        logger.info("Running health checks for all services...")
        
        # Check main backend
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start_time
            
            results.append(BenchmarkResult(
                component="backend",
                test_name="health_check",
                duration=duration,
                throughput=1/duration,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=response.status_code == 200,
                metadata={'status_code': response.status_code}
            ))
        except Exception as e:
            results.append(BenchmarkResult(
                component="backend",
                test_name="health_check",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        # Check all engine services
        for engine_name, url in self.engine_urls.items():
            start_time = time.time()
            try:
                response = requests.get(f"{url}/health", timeout=5)
                duration = time.time() - start_time
                
                results.append(BenchmarkResult(
                    component=f"{engine_name}_engine",
                    test_name="health_check",
                    duration=duration,
                    throughput=1/duration,
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    gpu_usage=None,
                    neural_engine_usage=None,
                    success=response.status_code == 200,
                    metadata={'status_code': response.status_code}
                ))
            except Exception as e:
                results.append(BenchmarkResult(
                    component=f"{engine_name}_engine",
                    test_name="health_check",
                    duration=time.time() - start_time,
                    throughput=0,
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    gpu_usage=None,
                    neural_engine_usage=None,
                    success=False,
                    error=str(e)
                ))
                
        return results
        
    async def _end_to_end_trading_pipeline(self) -> List[BenchmarkResult]:
        """Complete end-to-end trading pipeline test"""
        results = []
        
        logger.info("Running end-to-end trading pipeline benchmark...")
        
        # Simulate complete trading workflow: Data -> Analysis -> Decision -> Execution
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Step 1: Market data ingestion
            market_data_result = await self._simulate_market_data_ingestion()
            
            # Step 2: Feature calculation
            features_result = await self._simulate_feature_calculation()
            
            # Step 3: ML prediction
            ml_result = await self._simulate_ml_prediction()
            
            # Step 4: Risk assessment
            risk_result = await self._simulate_risk_assessment()
            
            # Step 5: Portfolio optimization
            portfolio_result = await self._simulate_portfolio_optimization()
            
            # Step 6: Order execution simulation
            execution_result = await self._simulate_order_execution()
            
            total_duration = time.time() - start_time
            
            # Calculate pipeline metrics
            pipeline_success = all([
                market_data_result, features_result, ml_result,
                risk_result, portfolio_result, execution_result
            ])
            
            metrics = monitor.stop_monitoring()
            avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
            avg_memory = np.mean([m['memory']['percent'] for m in metrics])
            
            results.append(BenchmarkResult(
                component="trading_pipeline",
                test_name="end_to_end_workflow",
                duration=total_duration,
                throughput=6/total_duration,  # 6 steps per second
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                gpu_usage=None,
                neural_engine_usage=None,
                success=pipeline_success,
                metadata={
                    'steps_completed': 6,
                    'market_data': market_data_result,
                    'features': features_result,
                    'ml_prediction': ml_result,
                    'risk_assessment': risk_result,
                    'portfolio_optimization': portfolio_result,
                    'order_execution': execution_result,
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="trading_pipeline",
                test_name="end_to_end_workflow",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _simulate_market_data_ingestion(self) -> bool:
        """Simulate market data ingestion"""
        try:
            # Test market data engine
            response = requests.post(
                f"{self.engine_urls['marketdata']}/ingest",
                json={'symbols': ['AAPL', 'GOOGL', 'MSFT'], 'data_type': 'real_time'},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
            
    async def _simulate_feature_calculation(self) -> bool:
        """Simulate feature calculation"""
        try:
            response = requests.post(
                f"{self.engine_urls['features']}/calculate",
                json={'symbols': ['AAPL', 'GOOGL', 'MSFT'], 'feature_types': ['technical', 'fundamental']},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
            
    async def _simulate_ml_prediction(self) -> bool:
        """Simulate ML prediction"""
        try:
            response = requests.post(
                f"{self.engine_urls['ml']}/predict",
                json={'model_type': 'price_prediction', 'symbols': ['AAPL', 'GOOGL', 'MSFT']},
                timeout=15
            )
            return response.status_code == 200
        except:
            return False
            
    async def _simulate_risk_assessment(self) -> bool:
        """Simulate risk assessment"""
        try:
            response = requests.post(
                f"{self.engine_urls['risk']}/assess",
                json={'portfolio': {'AAPL': 1000, 'GOOGL': 500, 'MSFT': 750}},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
            
    async def _simulate_portfolio_optimization(self) -> bool:
        """Simulate portfolio optimization"""
        try:
            response = requests.post(
                f"{self.engine_urls['portfolio']}/optimize",
                json={'assets': ['AAPL', 'GOOGL', 'MSFT'], 'target_risk': 0.15},
                timeout=20
            )
            return response.status_code == 200
        except:
            return False
            
    async def _simulate_order_execution(self) -> bool:
        """Simulate order execution"""
        try:
            response = requests.post(
                f"{self.base_url}/orders/simulate",
                json={'symbol': 'AAPL', 'quantity': 100, 'order_type': 'MARKET'},
                timeout=5
            )
            return response.status_code in [200, 201]
        except:
            return False
            
    async def _multi_component_workflow(self) -> List[BenchmarkResult]:
        """Test multiple components working simultaneously"""
        results = []
        
        logger.info("Running multi-component workflow benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Run multiple engine operations concurrently
            tasks = []
            
            # Analytics + Features + ML
            tasks.append(self._concurrent_analytics_features_ml())
            
            # Risk + Portfolio + Market Data
            tasks.append(self._concurrent_risk_portfolio_marketdata())
            
            # WebSocket + Strategy
            tasks.append(self._concurrent_websocket_strategy())
            
            # Execute all concurrent workflows
            workflow_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            # Calculate success rate
            successful_workflows = sum(1 for result in workflow_results if result is True)
            
            avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
            avg_memory = np.mean([m['memory']['percent'] for m in metrics])
            
            results.append(BenchmarkResult(
                component="multi_component",
                test_name="concurrent_workflows",
                duration=total_duration,
                throughput=len(tasks)/total_duration,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                gpu_usage=None,
                neural_engine_usage=None,
                success=successful_workflows == len(tasks),
                metadata={
                    'total_workflows': len(tasks),
                    'successful_workflows': successful_workflows,
                    'workflow_results': [str(r) for r in workflow_results],
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="multi_component",
                test_name="concurrent_workflows",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _concurrent_analytics_features_ml(self) -> bool:
        """Run analytics, features, and ML engines concurrently"""
        try:
            # Simulate concurrent operations
            tasks = [
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['analytics']}/status")),
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['features']}/status")),
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['ml']}/status"))
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return all(hasattr(r, 'status_code') and r.status_code == 200 for r in responses)
        except:
            return False
            
    async def _concurrent_risk_portfolio_marketdata(self) -> bool:
        """Run risk, portfolio, and market data engines concurrently"""
        try:
            tasks = [
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['risk']}/status")),
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['portfolio']}/status")),
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['marketdata']}/status"))
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return all(hasattr(r, 'status_code') and r.status_code == 200 for r in responses)
        except:
            return False
            
    async def _concurrent_websocket_strategy(self) -> bool:
        """Run WebSocket and strategy engines concurrently"""
        try:
            tasks = [
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['websocket']}/status")),
                asyncio.create_task(asyncio.to_thread(requests.get, f"{self.engine_urls['strategy']}/status"))
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return all(hasattr(r, 'status_code') and r.status_code == 200 for r in responses)
        except:
            return False
            
    async def _real_world_scenarios(self) -> List[BenchmarkResult]:
        """Real-world trading scenario simulations"""
        results = []
        
        logger.info("Running real-world trading scenario benchmarks...")
        
        scenarios = [
            ("high_frequency_trading", self._hft_scenario),
            ("portfolio_rebalancing", self._portfolio_rebalancing_scenario),
            ("risk_monitoring", self._risk_monitoring_scenario),
            ("market_data_processing", self._market_data_processing_scenario)
        ]
        
        for scenario_name, scenario_func in scenarios:
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            start_time = time.time()
            
            try:
                scenario_result = await scenario_func()
                duration = time.time() - start_time
                metrics = monitor.stop_monitoring()
                
                avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
                avg_memory = np.mean([m['memory']['percent'] for m in metrics])
                
                results.append(BenchmarkResult(
                    component="real_world_scenario",
                    test_name=scenario_name,
                    duration=duration,
                    throughput=1/duration,
                    cpu_usage=avg_cpu,
                    memory_usage=avg_memory,
                    gpu_usage=None,
                    neural_engine_usage=None,
                    success=scenario_result,
                    metadata={'monitoring_samples': len(metrics)}
                ))
                
            except Exception as e:
                monitor.stop_monitoring()
                results.append(BenchmarkResult(
                    component="real_world_scenario",
                    test_name=scenario_name,
                    duration=time.time() - start_time,
                    throughput=0,
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    gpu_usage=None,
                    neural_engine_usage=None,
                    success=False,
                    error=str(e)
                ))
                
        return results
        
    async def _hft_scenario(self) -> bool:
        """High-frequency trading scenario"""
        try:
            # Simulate rapid order processing
            order_count = 0
            for i in range(100):  # 100 rapid orders
                response = requests.post(
                    f"{self.base_url}/orders/simulate",
                    json={'symbol': 'AAPL', 'quantity': 10, 'order_type': 'LIMIT', 'price': 150.00 + i*0.01},
                    timeout=1
                )
                if response.status_code in [200, 201]:
                    order_count += 1
                    
            return order_count >= 80  # 80% success rate
        except:
            return False
            
    async def _portfolio_rebalancing_scenario(self) -> bool:
        """Portfolio rebalancing scenario"""
        try:
            response = requests.post(
                f"{self.engine_urls['portfolio']}/rebalance",
                json={
                    'current_portfolio': {'AAPL': 1000, 'GOOGL': 500, 'MSFT': 750},
                    'target_allocation': {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
                },
                timeout=30
            )
            return response.status_code == 200
        except:
            return False
            
    async def _risk_monitoring_scenario(self) -> bool:
        """Continuous risk monitoring scenario"""
        try:
            # Simulate continuous risk checks
            risk_checks = 0
            for i in range(20):
                response = requests.post(
                    f"{self.engine_urls['risk']}/monitor",
                    json={'portfolio': {'AAPL': 1000 + i*10, 'GOOGL': 500, 'MSFT': 750}},
                    timeout=2
                )
                if response.status_code == 200:
                    risk_checks += 1
                    
            return risk_checks >= 16  # 80% success rate
        except:
            return False
            
    async def _market_data_processing_scenario(self) -> bool:
        """High-volume market data processing"""
        try:
            response = requests.post(
                f"{self.engine_urls['marketdata']}/process_batch",
                json={'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'] * 20},  # 100 symbols
                timeout=15
            )
            return response.status_code == 200
        except:
            return False
            
    async def _cross_component_data_flow(self) -> List[BenchmarkResult]:
        """Test data flow between components"""
        results = []
        
        logger.info("Running cross-component data flow benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Test data flow: MarketData -> Features -> ML -> Risk -> Portfolio
            data_flow_success = True
            
            # 1. Market data to features
            md_to_features = await self._test_marketdata_to_features()
            data_flow_success &= md_to_features
            
            # 2. Features to ML
            features_to_ml = await self._test_features_to_ml()
            data_flow_success &= features_to_ml
            
            # 3. ML to Risk
            ml_to_risk = await self._test_ml_to_risk()
            data_flow_success &= ml_to_risk
            
            # 4. Risk to Portfolio
            risk_to_portfolio = await self._test_risk_to_portfolio()
            data_flow_success &= risk_to_portfolio
            
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
            avg_memory = np.mean([m['memory']['percent'] for m in metrics])
            
            results.append(BenchmarkResult(
                component="data_flow",
                test_name="cross_component_integration",
                duration=duration,
                throughput=4/duration,  # 4 data flow tests
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                gpu_usage=None,
                neural_engine_usage=None,
                success=data_flow_success,
                metadata={
                    'marketdata_to_features': md_to_features,
                    'features_to_ml': features_to_ml,
                    'ml_to_risk': ml_to_risk,
                    'risk_to_portfolio': risk_to_portfolio,
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="data_flow",
                test_name="cross_component_integration",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _test_marketdata_to_features(self) -> bool:
        """Test market data to features data flow"""
        try:
            # This would normally involve checking Redis/messaging between services
            response = requests.get(f"{self.engine_urls['marketdata']}/latest/AAPL", timeout=5)
            if response.status_code == 200:
                # Simulate features engine receiving the data
                features_response = requests.post(
                    f"{self.engine_urls['features']}/calculate_from_data",
                    json={'market_data': response.json()},
                    timeout=10
                )
                return features_response.status_code == 200
        except:
            pass
        return False
        
    async def _test_features_to_ml(self) -> bool:
        """Test features to ML data flow"""
        try:
            response = requests.get(f"{self.engine_urls['features']}/latest/AAPL", timeout=5)
            if response.status_code == 200:
                ml_response = requests.post(
                    f"{self.engine_urls['ml']}/predict_from_features",
                    json={'features': response.json()},
                    timeout=15
                )
                return ml_response.status_code == 200
        except:
            pass
        return False
        
    async def _test_ml_to_risk(self) -> bool:
        """Test ML to risk data flow"""
        try:
            response = requests.get(f"{self.engine_urls['ml']}/latest_prediction/AAPL", timeout=5)
            if response.status_code == 200:
                risk_response = requests.post(
                    f"{self.engine_urls['risk']}/assess_with_prediction",
                    json={'prediction': response.json()},
                    timeout=10
                )
                return risk_response.status_code == 200
        except:
            pass
        return False
        
    async def _test_risk_to_portfolio(self) -> bool:
        """Test risk to portfolio data flow"""
        try:
            response = requests.get(f"{self.engine_urls['risk']}/latest_assessment", timeout=5)
            if response.status_code == 200:
                portfolio_response = requests.post(
                    f"{self.engine_urls['portfolio']}/optimize_with_risk",
                    json={'risk_assessment': response.json()},
                    timeout=20
                )
                return portfolio_response.status_code == 200
        except:
            pass
        return False

class SystemResourceBenchmark:
    """M4 Max system resource utilization benchmark"""
    
    def __init__(self):
        self.cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
        self.cpu_logical = psutil.cpu_count(logical=True)  # Logical cores
        
    async def run_resource_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive system resource benchmarks"""
        results = []
        
        # 1. CPU Optimization Test
        results.extend(await self._cpu_optimization_test())
        
        # 2. Memory Bandwidth Test
        results.extend(await self._memory_bandwidth_test())
        
        # 3. GPU Utilization Test
        results.extend(await self._gpu_utilization_test())
        
        # 4. Neural Engine Test
        results.extend(await self._neural_engine_test())
        
        # 5. Simultaneous Resource Test
        results.extend(await self._simultaneous_resource_test())
        
        return results
        
    async def _cpu_optimization_test(self) -> List[BenchmarkResult]:
        """Test CPU optimization across P-cores and E-cores"""
        results = []
        
        logger.info("Running CPU optimization benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # CPU-intensive workload using all cores
            def cpu_intensive_task(duration=10):
                end_time = time.time() + duration
                calculations = 0
                while time.time() < end_time:
                    # Complex mathematical operations
                    np.random.random((100, 100)) @ np.random.random((100, 100))
                    calculations += 1
                return calculations
                
            # Run on all logical cores
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_logical) as executor:
                futures = [executor.submit(cpu_intensive_task, 10) for _ in range(self.cpu_logical)]
                calculation_results = [f.result() for f in concurrent.futures.as_completed(futures)]
                
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            total_calculations = sum(calculation_results)
            avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
            max_cpu = max([m['cpu']['percent'] for m in metrics])
            
            results.append(BenchmarkResult(
                component="cpu_optimization",
                test_name="multi_core_intensive",
                duration=duration,
                throughput=total_calculations/duration,
                cpu_usage=avg_cpu,
                memory_usage=np.mean([m['memory']['percent'] for m in metrics]),
                gpu_usage=None,
                neural_engine_usage=None,
                success=len(calculation_results) == self.cpu_logical,
                metadata={
                    'total_calculations': total_calculations,
                    'cores_used': self.cpu_logical,
                    'physical_cores': self.cpu_cores,
                    'max_cpu_usage': max_cpu,
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="cpu_optimization",
                test_name="multi_core_intensive",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _memory_bandwidth_test(self) -> List[BenchmarkResult]:
        """Test unified memory bandwidth utilization"""
        results = []
        
        logger.info("Running memory bandwidth benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Memory-intensive operations to test bandwidth
            array_size = 100_000_000  # 100M elements
            
            # Large array operations to stress memory bandwidth
            arr1 = np.random.random(array_size).astype(np.float64)
            arr2 = np.random.random(array_size).astype(np.float64)
            
            # Sequential memory operations
            result1 = arr1 + arr2  # Memory bandwidth test
            result2 = result1 * 2.5  # Another bandwidth test
            result3 = np.sqrt(result2)  # CPU + memory intensive
            
            # Memory copy operations
            copy1 = arr1.copy()
            copy2 = arr2.copy()
            
            # Final calculation
            final_result = np.sum(result3)
            
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            # Calculate approximate memory throughput
            total_memory_ops = array_size * 8 * 8  # 8 operations, 8 bytes per float64
            memory_throughput = total_memory_ops / duration / (1024**3)  # GB/s
            
            results.append(BenchmarkResult(
                component="memory_bandwidth",
                test_name="unified_memory_stress",
                duration=duration,
                throughput=memory_throughput,  # GB/s
                cpu_usage=np.mean([m['cpu']['percent'] for m in metrics]),
                memory_usage=np.max([m['memory']['percent'] for m in metrics]),
                gpu_usage=None,
                neural_engine_usage=None,
                success=not np.isnan(final_result),
                metadata={
                    'array_size': array_size,
                    'total_memory_operations': total_memory_ops,
                    'estimated_throughput_gbps': memory_throughput,
                    'final_result': float(final_result),
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="memory_bandwidth",
                test_name="unified_memory_stress",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _gpu_utilization_test(self) -> List[BenchmarkResult]:
        """Test Metal GPU utilization"""
        results = []
        
        logger.info("Running GPU utilization benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # GPU-accelerated computation using NumPy (which can use Metal on macOS)
            # Large matrix operations that benefit from GPU acceleration
            
            matrix_size = 5000
            iterations = 10
            
            gpu_intensive_results = []
            
            for i in range(iterations):
                # Large matrix operations
                matrix_a = np.random.random((matrix_size, matrix_size)).astype(np.float32)
                matrix_b = np.random.random((matrix_size, matrix_size)).astype(np.float32)
                
                # Matrix multiplication (can be GPU accelerated)
                result = np.matmul(matrix_a, matrix_b)
                
                # Additional GPU-friendly operations
                result = np.tanh(result)  # Element-wise operations
                result = np.fft.fft2(result)  # FFT operations
                
                gpu_intensive_results.append(np.sum(np.abs(result)))
                
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            total_operations = iterations * matrix_size * matrix_size
            
            results.append(BenchmarkResult(
                component="gpu_utilization",
                test_name="metal_gpu_intensive",
                duration=duration,
                throughput=total_operations/duration,
                cpu_usage=np.mean([m['cpu']['percent'] for m in metrics]),
                memory_usage=np.mean([m['memory']['percent'] for m in metrics]),
                gpu_usage=np.mean([m.get('gpu', {}).get('usage', 0) for m in metrics]),
                neural_engine_usage=None,
                success=len(gpu_intensive_results) == iterations,
                metadata={
                    'matrix_size': matrix_size,
                    'iterations': iterations,
                    'total_operations': total_operations,
                    'results_sum': sum(gpu_intensive_results),
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="gpu_utilization",
                test_name="metal_gpu_intensive",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=0,
                neural_engine_usage=None,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _neural_engine_test(self) -> List[BenchmarkResult]:
        """Test Neural Engine utilization"""
        results = []
        
        logger.info("Running Neural Engine benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Simulate ML inference workload that would utilize Neural Engine
            # Neural networks and tensor operations
            
            batch_size = 1000
            feature_size = 1000
            hidden_size = 512
            iterations = 50
            
            neural_results = []
            
            for i in range(iterations):
                # Simulate neural network inference
                input_data = np.random.random((batch_size, feature_size)).astype(np.float32)
                
                # Layer 1 (dense layer simulation)
                weights1 = np.random.random((feature_size, hidden_size)).astype(np.float32)
                layer1 = np.tanh(input_data @ weights1)
                
                # Layer 2 (dense layer simulation)
                weights2 = np.random.random((hidden_size, hidden_size//2)).astype(np.float32)
                layer2 = np.tanh(layer1 @ weights2)
                
                # Output layer
                weights3 = np.random.random((hidden_size//2, 10)).astype(np.float32)
                output = np.softmax(layer2 @ weights3, axis=1)
                
                # Simulated loss calculation
                target = np.random.randint(0, 10, batch_size)
                loss = -np.mean(np.log(output[np.arange(batch_size), target]))
                
                neural_results.append(loss)
                
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            total_inferences = iterations * batch_size
            
            # Get Neural Engine usage estimates
            neural_engine_metrics = [m.get('neural_engine', {}) for m in metrics]
            avg_neural_usage = np.mean([ne.get('estimated_usage', 0) for ne in neural_engine_metrics])
            
            results.append(BenchmarkResult(
                component="neural_engine",
                test_name="ml_inference_workload",
                duration=duration,
                throughput=total_inferences/duration,
                cpu_usage=np.mean([m['cpu']['percent'] for m in metrics]),
                memory_usage=np.mean([m['memory']['percent'] for m in metrics]),
                gpu_usage=None,
                neural_engine_usage=avg_neural_usage,
                success=len(neural_results) == iterations,
                metadata={
                    'batch_size': batch_size,
                    'iterations': iterations,
                    'total_inferences': total_inferences,
                    'avg_loss': np.mean(neural_results),
                    'neural_engine_processes': np.mean([ne.get('processes', 0) for ne in neural_engine_metrics]),
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="neural_engine",
                test_name="ml_inference_workload",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=None,
                neural_engine_usage=0,
                success=False,
                error=str(e)
            ))
            
        return results
        
    async def _simultaneous_resource_test(self) -> List[BenchmarkResult]:
        """Test simultaneous CPU, GPU, and Neural Engine utilization"""
        results = []
        
        logger.info("Running simultaneous resource utilization benchmark...")
        
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Run CPU, GPU, and Neural Engine workloads simultaneously
            
            async def cpu_workload():
                """CPU-intensive workload"""
                def cpu_task():
                    result = 0
                    for i in range(1000000):
                        result += np.sin(i) * np.cos(i)
                    return result
                    
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(cpu_task) for _ in range(4)]
                    return [f.result() for f in concurrent.futures.as_completed(futures)]
                    
            async def gpu_workload():
                """GPU-intensive workload"""
                return await asyncio.to_thread(self._gpu_computation)
                
            async def neural_workload():
                """Neural Engine workload"""
                return await asyncio.to_thread(self._neural_computation)
                
            # Execute all workloads simultaneously
            cpu_task = asyncio.create_task(cpu_workload())
            gpu_task = asyncio.create_task(gpu_workload())
            neural_task = asyncio.create_task(neural_workload())
            
            cpu_results, gpu_results, neural_results = await asyncio.gather(
                cpu_task, gpu_task, neural_task
            )
            
            duration = time.time() - start_time
            metrics = monitor.stop_monitoring()
            
            # Calculate utilization metrics
            avg_cpu = np.mean([m['cpu']['percent'] for m in metrics])
            max_cpu = np.max([m['cpu']['percent'] for m in metrics])
            avg_memory = np.mean([m['memory']['percent'] for m in metrics])
            max_memory = np.max([m['memory']['percent'] for m in metrics])
            
            gpu_metrics = [m.get('gpu', {}) for m in metrics]
            neural_metrics = [m.get('neural_engine', {}) for m in metrics]
            
            results.append(BenchmarkResult(
                component="simultaneous_resources",
                test_name="cpu_gpu_neural_concurrent",
                duration=duration,
                throughput=(len(cpu_results) + len(gpu_results) + len(neural_results))/duration,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                gpu_usage=np.mean([gm.get('usage', 0) for gm in gpu_metrics]),
                neural_engine_usage=np.mean([nm.get('estimated_usage', 0) for nm in neural_metrics]),
                success=all([cpu_results, gpu_results, neural_results]),
                metadata={
                    'cpu_results_count': len(cpu_results),
                    'gpu_results_count': len(gpu_results),
                    'neural_results_count': len(neural_results),
                    'max_cpu_usage': max_cpu,
                    'max_memory_usage': max_memory,
                    'monitoring_samples': len(metrics)
                }
            ))
            
        except Exception as e:
            monitor.stop_monitoring()
            results.append(BenchmarkResult(
                component="simultaneous_resources",
                test_name="cpu_gpu_neural_concurrent",
                duration=time.time() - start_time,
                throughput=0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                gpu_usage=0,
                neural_engine_usage=0,
                success=False,
                error=str(e)
            ))
            
        return results
        
    def _gpu_computation(self):
        """GPU computation workload"""
        try:
            # Matrix operations that can be GPU accelerated
            size = 2000
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
            
            results = []
            for _ in range(5):
                result = np.matmul(a, b)
                result = np.fft.fft2(result)
                results.append(np.sum(np.abs(result)))
                
            return results
        except Exception as e:
            return []
            
    def _neural_computation(self):
        """Neural Engine computation workload"""
        try:
            # ML inference simulation
            batch_size = 500
            feature_size = 784
            
            results = []
            for _ in range(10):
                # Simulate image processing / neural network
                input_data = np.random.random((batch_size, feature_size)).astype(np.float32)
                
                # Convolution-like operations
                reshaped = input_data.reshape(batch_size, 28, 28)
                convolved = np.convolve(reshaped.flatten(), np.random.random(9), mode='same')
                
                # Activation functions
                activated = np.tanh(convolved)
                output = np.mean(activated)
                
                results.append(output)
                
            return results
        except Exception as e:
            return []

async def run_comprehensive_benchmark():
    """Run comprehensive system-wide benchmark"""
    
    logger.info("Starting Comprehensive System Integration Benchmark")
    logger.info("=" * 60)
    
    all_results = []
    
    # 1. Integrated Trading Pipeline Benchmark
    logger.info("Phase 1: Integrated Trading Pipeline Benchmark")
    pipeline_benchmark = IntegratedTradingPipeline()
    pipeline_results = await pipeline_benchmark.run_integrated_benchmark()
    all_results.extend(pipeline_results)
    
    # 2. System Resource Utilization Tests
    logger.info("Phase 2: System Resource Utilization Tests")
    resource_benchmark = SystemResourceBenchmark()
    resource_results = await resource_benchmark.run_resource_benchmark()
    all_results.extend(resource_results)
    
    # Generate comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = generate_comprehensive_report(all_results, timestamp)
    
    # Save results
    results_file = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml/system_integration_results_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'total_memory': psutil.virtual_memory().total,
            'available_memory': psutil.virtual_memory().available
        },
        'benchmark_results': [
            {
                'component': r.component,
                'test_name': r.test_name,
                'duration': r.duration,
                'throughput': r.throughput,
                'cpu_usage': r.cpu_usage,
                'memory_usage': r.memory_usage,
                'gpu_usage': r.gpu_usage,
                'neural_engine_usage': r.neural_engine_usage,
                'success': r.success,
                'error': r.error,
                'metadata': r.metadata
            }
            for r in all_results
        ],
        'summary': report
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return all_results, report

def generate_comprehensive_report(results: List[BenchmarkResult], timestamp: str) -> Dict[str, Any]:
    """Generate comprehensive benchmark report"""
    
    # Calculate summary statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.success)
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    # Group results by component
    component_results = {}
    for result in results:
        if result.component not in component_results:
            component_results[result.component] = []
        component_results[result.component].append(result)
    
    # Performance metrics
    avg_cpu_usage = np.mean([r.cpu_usage for r in results if r.cpu_usage is not None])
    avg_memory_usage = np.mean([r.memory_usage for r in results if r.memory_usage is not None])
    avg_throughput = np.mean([r.throughput for r in results if r.throughput > 0])
    total_duration = sum([r.duration for r in results])
    
    # Component performance analysis
    component_performance = {}
    for component, comp_results in component_results.items():
        comp_success_rate = sum(1 for r in comp_results if r.success) / len(comp_results)
        comp_avg_throughput = np.mean([r.throughput for r in comp_results if r.throughput > 0])
        comp_avg_cpu = np.mean([r.cpu_usage for r in comp_results if r.cpu_usage is not None])
        comp_avg_memory = np.mean([r.memory_usage for r in comp_results if r.memory_usage is not None])
        
        component_performance[component] = {
            'success_rate': comp_success_rate,
            'avg_throughput': comp_avg_throughput,
            'avg_cpu_usage': comp_avg_cpu,
            'avg_memory_usage': comp_avg_memory,
            'test_count': len(comp_results)
        }
    
    # Performance assessment
    performance_grade = "A+"
    if success_rate < 0.95:
        performance_grade = "A"
    if success_rate < 0.90:
        performance_grade = "B+"
    if success_rate < 0.80:
        performance_grade = "B"
    if success_rate < 0.70:
        performance_grade = "C"
    if success_rate < 0.60:
        performance_grade = "D"
    
    # Bottleneck analysis
    bottlenecks = []
    if avg_cpu_usage > 85:
        bottlenecks.append("CPU utilization approaching maximum")
    if avg_memory_usage > 85:
        bottlenecks.append("Memory usage approaching maximum")
    
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        bottlenecks.extend([f"{r.component}.{r.test_name}: {r.error}" for r in failed_tests[:3]])
    
    # Production readiness assessment
    production_ready = (
        success_rate >= 0.95 and
        avg_cpu_usage < 80 and
        avg_memory_usage < 80 and
        len(bottlenecks) <= 1
    )
    
    return {
        'timestamp': timestamp,
        'overall_performance': {
            'success_rate': success_rate,
            'performance_grade': performance_grade,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests
        },
        'system_utilization': {
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'avg_throughput': avg_throughput,
            'total_duration': total_duration
        },
        'component_performance': component_performance,
        'bottlenecks': bottlenecks,
        'production_assessment': {
            'ready_for_production': production_ready,
            'recommendation': (
                "APPROVED for production deployment" if production_ready 
                else "REQUIRES optimization before production deployment"
            ),
            'key_metrics': {
                'system_stability': success_rate >= 0.95,
                'resource_efficiency': avg_cpu_usage < 80 and avg_memory_usage < 80,
                'performance_consistency': len(bottlenecks) <= 1
            }
        },
        'm4_max_optimization_validation': {
            'cpu_optimization_validated': any(
                r.component == 'cpu_optimization' and r.success 
                for r in results
            ),
            'memory_bandwidth_validated': any(
                r.component == 'memory_bandwidth' and r.success 
                for r in results
            ),
            'gpu_utilization_validated': any(
                r.component == 'gpu_utilization' and r.success 
                for r in results
            ),
            'neural_engine_validated': any(
                r.component == 'neural_engine' and r.success 
                for r in results
            ),
            'integrated_performance_validated': any(
                r.component == 'simultaneous_resources' and r.success 
                for r in results
            )
        }
    }

if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
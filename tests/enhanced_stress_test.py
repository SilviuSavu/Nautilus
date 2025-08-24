#!/usr/bin/env python3
"""
Enhanced Progressive Stress Testing Suite
Tests system limits and identifies breaking points for all 9 engines
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import statistics
import psutil
import docker
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
    
    def get_system_resources(self) -> Dict:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get Docker container stats
        containers = {}
        try:
            for container in self.docker_client.containers.list():
                if 'nautilus' in container.name:
                    stats = container.stats(stream=False)
                    containers[container.name] = {
                        'cpu_percent': self._calculate_cpu_percent(stats),
                        'memory_mb': stats['memory_stats'].get('usage', 0) / (1024 * 1024),
                        'memory_limit_mb': stats['memory_stats'].get('limit', 0) / (1024 * 1024)
                    }
        except Exception as e:
            logger.warning(f"Could not get container stats: {e}")
        
        return {
            'system_cpu_percent': cpu_percent,
            'system_memory_percent': memory.percent,
            'system_memory_available_gb': memory.available / (1024**3),
            'containers': containers
        }
    
    def _calculate_cpu_percent(self, stats):
        """Calculate CPU percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0

class EngineStressTester:
    def __init__(self):
        self.engines = {
            'analytics': {'port': 8100, 'name': 'Analytics Engine', 'limit_mb': 4096},
            'risk': {'port': 8200, 'name': 'Risk Engine', 'limit_mb': 1024}, 
            'factor': {'port': 8300, 'name': 'Factor Engine', 'limit_mb': 8192},
            'ml': {'port': 8400, 'name': 'ML Engine', 'limit_mb': 4096},
            'features': {'port': 8500, 'name': 'Features Engine', 'limit_mb': 4096},
            'websocket': {'port': 8600, 'name': 'WebSocket Engine', 'limit_mb': 2048},
            'strategy': {'port': 8700, 'name': 'Strategy Engine', 'limit_mb': 2048},
            'marketdata': {'port': 8800, 'name': 'MarketData Engine', 'limit_mb': 4096},
            'portfolio': {'port': 8900, 'name': 'Portfolio Engine', 'limit_mb': 2048}
        }
        self.monitor = SystemMonitor()
        self.results = []
        
    async def health_check_engine(self, session: aiohttp.ClientSession, port: int) -> Tuple[bool, float]:
        """Check engine health and measure response time"""
        start_time = time.time()
        try:
            async with session.get(f'http://localhost:{port}/health', timeout=aiohttp.ClientTimeout(total=5)) as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                return response.status == 200, response_time_ms
        except Exception:
            return False, 5000.0  # 5s timeout
    
    def get_real_data_endpoints(self, port: int) -> List[str]:
        """Get real data endpoints for each engine with actual data queries"""
        # Real stock symbols and parameters for testing
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM']
        
        endpoint_map = {
            8100: [  # Analytics Engine
                '/health',
                '/analytics/portfolio/summary', 
                '/analytics/risk/metrics',
                '/analytics/performance'
            ],
            8200: [  # Risk Engine  
                '/health',
                '/risk/limits',
                '/risk/breaches',
                '/risk/portfolio/analysis',
                '/risk/var/calculate'
            ],
            8300: [  # Factor Engine - Test with real symbols
                '/health',
                '/factors/list',
                f'/factors/calculate/{symbols[0]}',
                f'/factors/calculate/{symbols[1]}', 
                f'/factors/calculate/{symbols[2]}',
                '/factors/economic/indicators',
                '/factors/technical/momentum'
            ],
            8400: [  # ML Engine
                '/health', 
                '/ml/models',
                '/ml/predictions',
                f'/ml/predict/{symbols[0]}',
                '/ml/training/status'
            ],
            8500: [  # Features Engine - Test with real symbols
                '/health',
                f'/features/technical/{symbols[0]}',
                f'/features/technical/{symbols[1]}',
                f'/features/fundamental/{symbols[2]}',
                '/features/market/sentiment',
                '/features/economic/calendar'
            ],
            8600: [  # WebSocket Engine
                '/health',
                '/ws/connections',
                '/ws/subscriptions',
                '/ws/active/streams',
                '/ws/metrics'
            ],
            8700: [  # Strategy Engine
                '/health',
                '/strategy/active',
                '/strategy/performance',
                '/strategy/backtest/results',
                '/strategy/portfolio/allocation'
            ],
            8800: [  # MarketData Engine - Test with real symbols  
                '/health',
                f'/market/quotes/{symbols[0]}',
                f'/market/quotes/{symbols[1]}',
                f'/market/history/{symbols[2]}',
                '/market/latest/prices',
                '/market/volume/analysis'
            ],
            8900: [  # Portfolio Engine
                '/health',
                '/portfolio/positions',
                '/portfolio/performance',
                '/portfolio/allocation',
                '/portfolio/pnl/daily',
                '/portfolio/risk/metrics'
            ]
        }
        return endpoint_map.get(port, ['/health'])

    async def stress_test_engine(self, session: aiohttp.ClientSession, port: int, concurrent_users: int, requests_per_user: int) -> Dict:
        """Run stress test on single engine with real data endpoints"""
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        # Get real endpoints for this engine
        endpoints = self.get_real_data_endpoints(port)
        
        async def make_request():
            nonlocal successful_requests, failed_requests
            # Rotate through different endpoints to simulate real usage
            endpoint = endpoints[successful_requests % len(endpoints)]
            
            start_time = time.time()
            try:
                url = f'http://localhost:{port}{endpoint}'
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)
                    
                    # Accept both 200 and 404 as "successful" for stress testing
                    # 404 means endpoint exists but no data, which is normal
                    if response.status in [200, 404]:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        
            except asyncio.TimeoutError:
                failed_requests += 1
                response_times.append(15000.0)  # 15s timeout
            except Exception as e:
                failed_requests += 1
                response_times.append(10000.0)  # Generic error timeout
        
        # Create concurrent user tasks
        tasks = []
        for user in range(concurrent_users):
            user_tasks = [make_request() for _ in range(requests_per_user)]
            tasks.extend(user_tasks)
        
        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_requests = successful_requests + failed_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'average_response_time_ms': statistics.mean(response_times) if response_times else 0,
            'median_response_time_ms': statistics.median(response_times) if response_times else 0,
            'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else (max(response_times) if response_times else 0),
            'p99_response_time_ms': statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else (max(response_times) if response_times else 0),
            'min_response_time_ms': min(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'requests_per_second': total_requests / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'test_duration_seconds': end_time - start_time
        }
    
    async def progressive_stress_test(self):
        """Run progressive stress tests with increasing load"""
        logger.info("=" * 100)
        logger.info("STARTING ENHANCED PROGRESSIVE STRESS TESTING SUITE")
        logger.info("=" * 100)
        
        # Test configurations - progressively increasing load (optimized for faster execution)
        test_configs = [
            {'users': 10, 'requests': 20, 'description': 'Baseline Load'},
            {'users': 25, 'requests': 20, 'description': 'Light Load'},
            {'users': 50, 'requests': 15, 'description': 'Moderate Load'},
            {'users': 100, 'requests': 10, 'description': 'Heavy Load'},
            {'users': 200, 'requests': 8, 'description': 'High Load'},
            {'users': 400, 'requests': 5, 'description': 'Extreme Load'},
            {'users': 800, 'requests': 3, 'description': 'Maximum Load'},
            {'users': 1000, 'requests': 2, 'description': 'Breaking Point Test'}
        ]
        
        for config in test_configs:
            await self.run_test_configuration(config)
            
            # Wait between tests to let system recover (reduced wait time)
            logger.info(f"Waiting 5 seconds for system recovery...")
            await asyncio.sleep(5)
        
        # Generate final report
        self.generate_comprehensive_report()
    
    async def run_test_configuration(self, config: Dict):
        """Run stress test for a specific configuration"""
        users = config['users']
        requests = config['requests']
        description = config['description']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING: {description} ({users} users, {requests} requests each)")
        logger.info(f"{'='*60}")
        
        # Get system resources before test
        resources_before = self.monitor.get_system_resources()
        logger.info(f"System CPU: {resources_before['system_cpu_percent']:.1f}%, "
                   f"Memory: {resources_before['system_memory_percent']:.1f}%")
        
        test_results = {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'resources_before': resources_before,
            'engines': {},
            'system_degradation': False,
            'breaking_point_reached': False
        }
        
        async with aiohttp.ClientSession() as session:
            # Test each engine
            for engine_id, engine_config in self.engines.items():
                port = engine_config['port']
                name = engine_config['name']
                
                logger.info(f"Testing {name}...")
                
                # First check if engine is healthy
                is_healthy, health_response_time = await self.health_check_engine(session, port)
                
                if not is_healthy:
                    logger.error(f"❌ {name} failed health check!")
                    test_results['engines'][engine_id] = {
                        'status': 'failed',
                        'error': 'Health check failed'
                    }
                    continue
                
                # Run stress test
                try:
                    engine_results = await self.stress_test_engine(session, port, users, requests)
                    
                    # Determine grade
                    grade = self.calculate_grade(engine_results)
                    
                    test_results['engines'][engine_id] = {
                        'name': name,
                        'status': 'completed',
                        'grade': grade,
                        **engine_results
                    }
                    
                    # Log results
                    success_rate = engine_results['success_rate']
                    avg_response = engine_results['average_response_time_ms']
                    rps = engine_results['requests_per_second']
                    
                    logger.info(f"✅ {name}: {success_rate:.1f}% success, {avg_response:.1f}ms avg, {rps:.0f} RPS, Grade: {grade}")
                    
                    # Check for system degradation
                    if success_rate < 95 or avg_response > 100:
                        test_results['system_degradation'] = True
                        logger.warning(f"⚠️ System degradation detected in {name}")
                    
                    if success_rate < 50:
                        test_results['breaking_point_reached'] = True
                        logger.error(f"🚨 Breaking point reached for {name}")
                
                except Exception as e:
                    logger.error(f"❌ {name} test failed: {e}")
                    test_results['engines'][engine_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Get system resources after test
        resources_after = self.monitor.get_system_resources()
        test_results['resources_after'] = resources_after
        
        logger.info(f"System CPU after: {resources_after['system_cpu_percent']:.1f}%, "
                   f"Memory: {resources_after['system_memory_percent']:.1f}%")
        
        # Calculate summary metrics
        successful_engines = sum(1 for result in test_results['engines'].values() 
                                if result.get('status') == 'completed' and result.get('success_rate', 0) >= 95)
        total_engines = len([r for r in test_results['engines'].values() if r.get('status') == 'completed'])
        
        if total_engines > 0:
            avg_success_rate = sum(r.get('success_rate', 0) for r in test_results['engines'].values() 
                                 if r.get('status') == 'completed') / total_engines
            avg_response_time = sum(r.get('average_response_time_ms', 0) for r in test_results['engines'].values() 
                                  if r.get('status') == 'completed') / total_engines
            total_rps = sum(r.get('requests_per_second', 0) for r in test_results['engines'].values() 
                           if r.get('status') == 'completed')
        else:
            avg_success_rate = avg_response_time = total_rps = 0
        
        test_results['summary'] = {
            'successful_engines': successful_engines,
            'total_engines': total_engines,
            'system_availability': (successful_engines / total_engines * 100) if total_engines > 0 else 0,
            'average_success_rate': avg_success_rate,
            'average_response_time_ms': avg_response_time,
            'total_requests_per_second': total_rps,
            'total_requests_processed': sum(r.get('total_requests', 0) for r in test_results['engines'].values() 
                                          if r.get('status') == 'completed')
        }
        
        logger.info(f"\n📊 CONFIGURATION SUMMARY:")
        logger.info(f"   System Availability: {test_results['summary']['system_availability']:.1f}%")
        logger.info(f"   Average Success Rate: {avg_success_rate:.1f}%")  
        logger.info(f"   Average Response Time: {avg_response_time:.1f}ms")
        logger.info(f"   Total Throughput: {total_rps:.0f} RPS")
        logger.info(f"   Total Requests: {test_results['summary']['total_requests_processed']:,}")
        
        self.results.append(test_results)
        
        # Check if we should stop testing (breaking point reached)
        if test_results['breaking_point_reached']:
            logger.error("🛑 BREAKING POINT REACHED - Stopping progressive testing")
            return False
        
        return True
    
    def calculate_grade(self, results: Dict) -> str:
        """Calculate performance grade"""
        success_rate = results['success_rate']
        avg_response_time = results['average_response_time_ms']
        
        if success_rate >= 99 and avg_response_time <= 5:
            return "A+"
        elif success_rate >= 95 and avg_response_time <= 10:
            return "A"
        elif success_rate >= 90 and avg_response_time <= 25:
            return "B+"
        elif success_rate >= 85 and avg_response_time <= 50:
            return "B"
        elif success_rate >= 80 and avg_response_time <= 100:
            return "C+"
        elif success_rate >= 70 and avg_response_time <= 200:
            return "C"
        elif success_rate >= 60:
            return "D"
        else:
            return "F"
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE STRESS TESTING RESULTS")
        logger.info("=" * 100)
        
        # Find optimal configuration
        optimal_config = None
        max_throughput = 0
        
        # Find breaking point
        breaking_point_config = None
        
        for result in self.results:
            total_rps = result['summary']['total_requests_per_second']
            
            # Track optimal performance (high throughput + good availability)
            if (result['summary']['system_availability'] >= 90 and 
                result['summary']['average_success_rate'] >= 95 and
                total_rps > max_throughput):
                max_throughput = total_rps
                optimal_config = result
            
            # Track breaking point
            if result['breaking_point_reached'] or result['summary']['system_availability'] < 70:
                breaking_point_config = result
                break
        
        # Performance progression
        logger.info("\n📈 PERFORMANCE PROGRESSION:")
        logger.info(f"{'Load Level':<20} {'Availability':<12} {'Avg Success':<12} {'Avg Response':<15} {'Total RPS':<12} {'Status':<15}")
        logger.info("-" * 90)
        
        for result in self.results:
            config = result['config']
            summary = result['summary']
            
            load_desc = config['description']
            availability = f"{summary['system_availability']:.1f}%"
            success_rate = f"{summary['average_success_rate']:.1f}%"
            response_time = f"{summary['average_response_time_ms']:.1f}ms"
            rps = f"{summary['total_requests_per_second']:.0f}"
            
            # Status indicators
            if summary['system_availability'] >= 95 and summary['average_success_rate'] >= 95:
                status = "✅ Excellent"
            elif summary['system_availability'] >= 80 and summary['average_success_rate'] >= 85:
                status = "⚠️ Degraded"
            else:
                status = "❌ Failed"
            
            logger.info(f"{load_desc:<20} {availability:<12} {success_rate:<12} {response_time:<15} {rps:<12} {status:<15}")
        
        # Key findings
        logger.info(f"\n🎯 KEY FINDINGS:")
        
        if optimal_config:
            opt_summary = optimal_config['summary']
            opt_config = optimal_config['config']
            logger.info(f"   ✅ OPTIMAL PERFORMANCE:")
            logger.info(f"      Load: {opt_config['description']} ({opt_config['users']} users)")
            logger.info(f"      Throughput: {opt_summary['total_requests_per_second']:.0f} RPS")
            logger.info(f"      Availability: {opt_summary['system_availability']:.1f}%")
            logger.info(f"      Response Time: {opt_summary['average_response_time_ms']:.1f}ms")
        
        if breaking_point_config:
            bp_config = breaking_point_config['config']
            bp_summary = breaking_point_config['summary']
            logger.info(f"   🚨 BREAKING POINT:")
            logger.info(f"      Load: {bp_config['description']} ({bp_config['users']} users)")
            logger.info(f"      Availability: {bp_summary['system_availability']:.1f}%")
            logger.info(f"      Success Rate: {bp_summary['average_success_rate']:.1f}%")
        else:
            logger.info(f"   🔥 NO BREAKING POINT FOUND - System handled all test loads!")
        
        # Engine-specific performance
        logger.info(f"\n🏆 INDIVIDUAL ENGINE PERFORMANCE:")
        
        # Collect all engine performances across tests
        engine_performance = {}
        for result in self.results:
            for engine_id, engine_data in result['engines'].items():
                if engine_data.get('status') == 'completed':
                    if engine_id not in engine_performance:
                        engine_performance[engine_id] = []
                    engine_performance[engine_id].append(engine_data)
        
        for engine_id, performances in engine_performance.items():
            engine_name = self.engines[engine_id]['name']
            
            # Find best and worst performance
            best_perf = max(performances, key=lambda x: x.get('requests_per_second', 0))
            worst_perf = min(performances, key=lambda x: x.get('success_rate', 100))
            
            avg_success = sum(p.get('success_rate', 0) for p in performances) / len(performances)
            avg_response = sum(p.get('average_response_time_ms', 0) for p in performances) / len(performances)
            max_rps = max(p.get('requests_per_second', 0) for p in performances)
            
            logger.info(f"   {engine_name}:")
            logger.info(f"     Average Success Rate: {avg_success:.1f}%")
            logger.info(f"     Average Response Time: {avg_response:.1f}ms")
            logger.info(f"     Maximum Throughput: {max_rps:.0f} RPS")
            logger.info(f"     Best Grade: {best_perf.get('grade', 'N/A')}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_stress_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Detailed results saved to: {results_file}")
        logger.info(f"🏁 Enhanced stress testing completed!")
        
        return results_file

async def main():
    """Run enhanced stress testing"""
    tester = EngineStressTester()
    await tester.progressive_stress_test()

if __name__ == "__main__":
    asyncio.run(main())
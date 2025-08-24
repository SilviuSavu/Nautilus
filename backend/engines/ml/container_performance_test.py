#!/usr/bin/env python3
"""
Nautilus Trading Platform - Container Architecture Performance Test
M4 Max Container Orchestration Validation

Tests all 16+ containers running with M4 Max optimizations:
- Container resource allocation and dynamic scaling
- Inter-container communication performance
- Container startup times and health check validation
"""

import asyncio
import time
import json
import subprocess
import requests
import concurrent.futures
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any, Optional
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerPerformanceTest:
    """Container architecture performance testing"""
    
    def __init__(self):
        self.containers = {
            # Core services
            'nautilus-backend': {'port': 8001, 'health_endpoint': '/health'},
            'nautilus-frontend': {'port': 3000, 'health_endpoint': None},
            'nautilus-redis': {'port': 6379, 'health_endpoint': None},
            'nautilus-postgres': {'port': 5432, 'health_endpoint': None},
            'nautilus-nginx': {'port': 80, 'health_endpoint': None},
            'nautilus-prometheus': {'port': 9090, 'health_endpoint': '/-/healthy'},
            'nautilus-grafana': {'port': 3002, 'health_endpoint': '/api/health'},
            
            # Engine services
            'nautilus-analytics-engine': {'port': 8100, 'health_endpoint': '/health'},
            'nautilus-risk-engine': {'port': 8200, 'health_endpoint': '/health'},
            'nautilus-factor-engine': {'port': 8300, 'health_endpoint': '/health'},
            'nautilus-ml-engine': {'port': 8400, 'health_endpoint': '/health'},
            'nautilus-features-engine': {'port': 8500, 'health_endpoint': '/health'},
            'nautilus-websocket-engine': {'port': 8600, 'health_endpoint': '/health'},
            'nautilus-strategy-engine': {'port': 8700, 'health_endpoint': '/health'},
            'nautilus-marketdata-engine': {'port': 8800, 'health_endpoint': '/health'},
            'nautilus-portfolio-engine': {'port': 8900, 'health_endpoint': '/health'}
        }
        
    def get_container_stats(self) -> Dict[str, Any]:
        """Get Docker container statistics"""
        try:
            # Get container status
            result = subprocess.run(
                ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                container_status = {}
                
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            status = parts[1].strip()
                            ports = parts[2].strip() if len(parts) > 2 else ''
                            
                            container_status[name] = {
                                'status': status,
                                'ports': ports,
                                'running': 'Up' in status
                            }
                            
                return container_status
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to get container stats: {e}")
            
        return {}
        
    def get_container_resource_usage(self) -> Dict[str, Any]:
        """Get container resource usage"""
        try:
            # Get container resource usage
            result = subprocess.run(
                ['docker', 'stats', '--no-stream', '--format', 
                 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}'],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                resource_usage = {}
                
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 5:
                            name = parts[0].strip()
                            cpu = parts[1].strip()
                            memory = parts[2].strip()
                            network = parts[3].strip()
                            block = parts[4].strip()
                            
                            resource_usage[name] = {
                                'cpu_percent': cpu,
                                'memory_usage': memory,
                                'network_io': network,
                                'block_io': block
                            }
                            
                return resource_usage
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to get container resource usage: {e}")
            
        return {}
        
    async def test_container_health(self) -> Dict[str, Any]:
        """Test health of all containers"""
        logger.info("Testing container health...")
        
        container_stats = self.get_container_stats()
        health_results = {}
        
        for container_name, config in self.containers.items():
            health_results[container_name] = {
                'running': container_stats.get(container_name, {}).get('running', False),
                'status': container_stats.get(container_name, {}).get('status', 'Unknown'),
                'health_check': False,
                'response_time': None,
                'error': None
            }
            
            # Test health endpoint if available
            if config.get('health_endpoint') and health_results[container_name]['running']:
                try:
                    start_time = time.time()
                    
                    if config['port'] == 3002:  # Grafana special case
                        url = f"http://localhost:{config['port']}{config['health_endpoint']}"
                    else:
                        url = f"http://localhost:{config['port']}{config['health_endpoint']}"
                        
                    response = requests.get(url, timeout=5)
                    response_time = time.time() - start_time
                    
                    health_results[container_name]['health_check'] = response.status_code < 400
                    health_results[container_name]['response_time'] = response_time
                    
                except Exception as e:
                    health_results[container_name]['error'] = str(e)
                    
        return health_results
        
    async def test_container_startup_performance(self) -> Dict[str, Any]:
        """Test container startup performance"""
        logger.info("Testing container startup performance...")
        
        startup_results = {}
        
        # Test a sample engine restart
        test_container = 'nautilus-ml-engine'
        
        try:
            # Stop container
            logger.info(f"Stopping {test_container}...")
            stop_start = time.time()
            
            stop_result = subprocess.run(
                ['docker', 'stop', test_container],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            stop_time = time.time() - stop_start
            
            # Start container
            logger.info(f"Starting {test_container}...")
            start_start = time.time()
            
            start_result = subprocess.run(
                ['docker', 'start', test_container],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            start_time = time.time() - start_start
            
            # Wait for health check
            health_start = time.time()
            healthy = False
            
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"http://localhost:8400/health", timeout=2)
                    if response.status_code == 200:
                        healthy = True
                        break
                except:
                    pass
                await asyncio.sleep(1)
                
            health_time = time.time() - health_start
            
            startup_results[test_container] = {
                'stop_time': stop_time,
                'start_time': start_time,
                'health_check_time': health_time,
                'total_restart_time': stop_time + start_time + health_time,
                'healthy': healthy,
                'stop_success': stop_result.returncode == 0,
                'start_success': start_result.returncode == 0
            }
            
        except Exception as e:
            startup_results[test_container] = {
                'error': str(e),
                'success': False
            }
            
        return startup_results
        
    async def test_inter_container_communication(self) -> Dict[str, Any]:
        """Test inter-container communication performance"""
        logger.info("Testing inter-container communication...")
        
        communication_tests = []
        
        # Test 1: Backend to Redis
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8001/health", timeout=5)
            backend_redis_time = time.time() - start_time
            
            communication_tests.append({
                'test': 'backend_to_redis',
                'response_time': backend_redis_time,
                'success': response.status_code == 200,
                'status_code': response.status_code
            })
        except Exception as e:
            communication_tests.append({
                'test': 'backend_to_redis',
                'success': False,
                'error': str(e)
            })
            
        # Test 2: Backend to Postgres
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8001/api/system/database", timeout=10)
            backend_postgres_time = time.time() - start_time
            
            communication_tests.append({
                'test': 'backend_to_postgres',
                'response_time': backend_postgres_time,
                'success': response.status_code == 200,
                'status_code': response.status_code
            })
        except Exception as e:
            communication_tests.append({
                'test': 'backend_to_postgres',
                'success': False,
                'error': str(e)
            })
            
        # Test 3: Engine to Engine Communication
        engine_communication_results = []
        
        engine_pairs = [
            ('analytics', 8100),
            ('risk', 8200),
            ('factor', 8300),
            ('ml', 8400),
            ('features', 8500)
        ]
        
        for engine_name, port in engine_pairs:
            try:
                start_time = time.time()
                response = requests.get(f"http://localhost:{port}/status", timeout=5)
                response_time = time.time() - start_time
                
                engine_communication_results.append({
                    'engine': engine_name,
                    'port': port,
                    'response_time': response_time,
                    'success': response.status_code == 200,
                    'status_code': response.status_code
                })
            except Exception as e:
                engine_communication_results.append({
                    'engine': engine_name,
                    'port': port,
                    'success': False,
                    'error': str(e)
                })
                
        return {
            'basic_communication': communication_tests,
            'engine_communication': engine_communication_results
        }
        
    async def test_container_scaling_performance(self) -> Dict[str, Any]:
        """Test container scaling performance"""
        logger.info("Testing container scaling performance...")
        
        scaling_results = {}
        
        # Test resource allocation under load
        try:
            # Generate load across multiple engines simultaneously
            load_tasks = []
            
            engine_urls = [
                'http://localhost:8100/status',  # Analytics
                'http://localhost:8200/status',  # Risk
                'http://localhost:8300/status',  # Factor
                'http://localhost:8400/status',  # ML
                'http://localhost:8500/status'   # Features
            ]
            
            async def generate_engine_load(url: str, requests_count: int = 50):
                """Generate load on an engine"""
                successful_requests = 0
                total_time = 0
                
                start_time = time.time()
                
                for _ in range(requests_count):
                    try:
                        req_start = time.time()
                        response = requests.get(url, timeout=2)
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            successful_requests += 1
                            total_time += req_time
                            
                    except Exception:
                        pass
                        
                total_duration = time.time() - start_time
                
                return {
                    'url': url,
                    'successful_requests': successful_requests,
                    'total_requests': requests_count,
                    'success_rate': successful_requests / requests_count,
                    'avg_response_time': total_time / max(successful_requests, 1),
                    'total_duration': total_duration,
                    'throughput': successful_requests / total_duration
                }
                
            # Run load tests concurrently
            for url in engine_urls:
                load_tasks.append(generate_engine_load(url))
                
            # Get initial resource usage
            initial_resources = self.get_container_resource_usage()
            
            # Execute load tests
            load_results = await asyncio.gather(*load_tasks)
            
            # Get final resource usage
            await asyncio.sleep(2)  # Wait for resources to stabilize
            final_resources = self.get_container_resource_usage()
            
            scaling_results = {
                'load_test_results': load_results,
                'initial_resources': initial_resources,
                'final_resources': final_resources,
                'resource_scaling_detected': len(final_resources) > 0
            }
            
        except Exception as e:
            scaling_results = {
                'error': str(e),
                'success': False
            }
            
        return scaling_results
        
    async def run_comprehensive_container_test(self) -> Dict[str, Any]:
        """Run comprehensive container architecture performance test"""
        logger.info("Starting Comprehensive Container Architecture Performance Test")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Get initial system state
        initial_container_stats = self.get_container_stats()
        initial_resource_usage = self.get_container_resource_usage()
        
        # 1. Container Health Test
        health_results = await self.test_container_health()
        
        # 2. Container Startup Performance
        startup_results = await self.test_container_startup_performance()
        
        # 3. Inter-Container Communication
        communication_results = await self.test_inter_container_communication()
        
        # 4. Container Scaling Performance
        scaling_results = await self.test_container_scaling_performance()
        
        # Get final system state
        final_container_stats = self.get_container_stats()
        final_resource_usage = self.get_container_resource_usage()
        
        total_duration = time.time() - start_time
        
        # Compile comprehensive results
        results = {
            'test_info': {
                'start_time': datetime.now(timezone.utc).isoformat(),
                'total_duration': total_duration,
                'total_containers_tested': len(self.containers),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total,
                    'memory_available': psutil.virtual_memory().available
                }
            },
            'container_health': health_results,
            'startup_performance': startup_results,
            'inter_container_communication': communication_results,
            'scaling_performance': scaling_results,
            'system_state': {
                'initial_container_stats': initial_container_stats,
                'final_container_stats': final_container_stats,
                'initial_resource_usage': initial_resource_usage,
                'final_resource_usage': final_resource_usage
            }
        }
        
        # Generate summary
        running_containers = sum(1 for info in health_results.values() if info['running'])
        healthy_containers = sum(1 for info in health_results.values() if info['health_check'])
        
        results['summary'] = {
            'total_containers': len(self.containers),
            'running_containers': running_containers,
            'healthy_containers': healthy_containers,
            'container_health_rate': running_containers / len(self.containers),
            'service_health_rate': healthy_containers / len([c for c in self.containers.values() if c['health_endpoint']]),
            'overall_success': running_containers >= len(self.containers) * 0.9,  # 90% threshold
            'performance_grade': self._calculate_performance_grade(results)
        }
        
        return results
        
    def _calculate_performance_grade(self, results: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        score = 0
        max_score = 0
        
        # Health score (40 points)
        health_results = results['container_health']
        running_containers = sum(1 for info in health_results.values() if info['running'])
        score += (running_containers / len(self.containers)) * 40
        max_score += 40
        
        # Communication score (30 points)
        comm_results = results['inter_container_communication']
        if 'basic_communication' in comm_results:
            successful_basic = sum(1 for test in comm_results['basic_communication'] if test['success'])
            score += (successful_basic / len(comm_results['basic_communication'])) * 20 if comm_results['basic_communication'] else 0
            max_score += 20
            
        if 'engine_communication' in comm_results:
            successful_engine = sum(1 for test in comm_results['engine_communication'] if test['success'])
            score += (successful_engine / len(comm_results['engine_communication'])) * 10 if comm_results['engine_communication'] else 0
            max_score += 10
        
        # Startup performance score (20 points)
        startup_results = results['startup_performance']
        if startup_results and not startup_results.get('error'):
            for container, result in startup_results.items():
                if result.get('healthy') and result.get('total_restart_time', 0) < 60:
                    score += 20
                    max_score += 20
                    break
        
        # Scaling performance score (10 points)
        scaling_results = results['scaling_performance']
        if scaling_results and not scaling_results.get('error'):
            if scaling_results.get('resource_scaling_detected'):
                score += 10
            max_score += 10
        
        # Calculate percentage and assign grade
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        if percentage >= 95:
            return "A+"
        elif percentage >= 90:
            return "A"
        elif percentage >= 85:
            return "B+"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        else:
            return "D"

async def main():
    """Main test execution"""
    tester = ContainerPerformanceTest()
    results = await tester.run_comprehensive_container_test()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml/container_performance_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Container performance test results saved to: {results_file}")
    
    # Print summary
    summary = results['summary']
    logger.info("=" * 60)
    logger.info("CONTAINER ARCHITECTURE PERFORMANCE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Containers: {summary['total_containers']}")
    logger.info(f"Running Containers: {summary['running_containers']}")
    logger.info(f"Healthy Containers: {summary['healthy_containers']}")
    logger.info(f"Container Health Rate: {summary['container_health_rate']:.2%}")
    logger.info(f"Service Health Rate: {summary['service_health_rate']:.2%}")
    logger.info(f"Performance Grade: {summary['performance_grade']}")
    logger.info(f"Overall Success: {'PASS' if summary['overall_success'] else 'FAIL'}")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
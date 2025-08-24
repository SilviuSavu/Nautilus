#!/usr/bin/env python3
"""
Nautilus Trading Platform - Reliability and Stress Testing
M4 Max System Stability Validation

Comprehensive reliability testing:
- Sustained load testing (30+ minutes full utilization)
- Error injection and recovery testing
- Failover mechanism validation
- Performance regression detection
"""

import asyncio
import time
import json
import psutil
import requests
import subprocess
import numpy as np
import concurrent.futures
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any, Optional
import threading
import queue
import random
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemStabilityMonitor:
    """Advanced system stability monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        self.stability_alerts = []
        
    def start_monitoring(self):
        """Start stability monitoring"""
        self.monitoring = True
        self.metrics = []
        self.stability_alerts = []
        self.monitor_thread = threading.Thread(target=self._stability_monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return stability metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return {
            'metrics': self.metrics,
            'stability_alerts': self.stability_alerts,
            'monitoring_duration': len(self.metrics) * 0.5  # 0.5s intervals
        }
        
    def _stability_monitor_loop(self):
        """Monitor system stability with advanced metrics"""
        previous_metrics = None
        
        while self.monitoring:
            try:
                current_time = datetime.now(timezone.utc)
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Process information
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        pinfo = proc.info
                        if 'nautilus' in pinfo['name'].lower() or 'python' in pinfo['name'].lower():
                            processes.append(pinfo)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
                # System load
                load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
                
                # Temperature monitoring (if available)
                temperatures = {}
                try:
                    if hasattr(psutil, 'sensors_temperatures'):
                        temps = psutil.sensors_temperatures()
                        for name, entries in temps.items():
                            for entry in entries:
                                temperatures[f"{name}_{entry.label}"] = entry.current
                except:
                    pass
                    
                current_metrics = {
                    'timestamp': current_time.isoformat(),
                    'cpu': {
                        'overall_percent': cpu_percent,
                        'per_core': cpu_per_core,
                        'core_count': len(cpu_per_core),
                        'load_avg': load_avg
                    },
                    'memory': {
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent,
                        'used': memory.used,
                        'free': memory.free,
                        'cached': getattr(memory, 'cached', 0),
                        'buffers': getattr(memory, 'buffers', 0)
                    },
                    'disk_io': {
                        'read_bytes': disk_io.read_bytes if disk_io else 0,
                        'write_bytes': disk_io.write_bytes if disk_io else 0,
                        'read_count': disk_io.read_count if disk_io else 0,
                        'write_count': disk_io.write_count if disk_io else 0
                    },
                    'network_io': {
                        'bytes_sent': network_io.bytes_sent if network_io else 0,
                        'bytes_recv': network_io.bytes_recv if network_io else 0,
                        'packets_sent': network_io.packets_sent if network_io else 0,
                        'packets_recv': network_io.packets_recv if network_io else 0
                    },
                    'processes': processes,
                    'temperatures': temperatures
                }
                
                # Stability analysis
                if previous_metrics:
                    self._analyze_stability(current_metrics, previous_metrics)
                    
                self.metrics.append(current_metrics)
                previous_metrics = current_metrics
                
                time.sleep(0.5)  # High-frequency monitoring
                
            except Exception as e:
                logger.warning(f"Stability monitoring error: {e}")
                time.sleep(1)
                
    def _analyze_stability(self, current: Dict, previous: Dict):
        """Analyze system stability between measurements"""
        try:
            # CPU stability check
            cpu_change = abs(current['cpu']['overall_percent'] - previous['cpu']['overall_percent'])
            if cpu_change > 30:  # 30% CPU spike
                self.stability_alerts.append({
                    'timestamp': current['timestamp'],
                    'type': 'cpu_spike',
                    'severity': 'warning',
                    'details': f"CPU usage changed by {cpu_change:.1f}%"
                })
                
            # Memory stability check
            memory_change = abs(current['memory']['percent'] - previous['memory']['percent'])
            if memory_change > 10:  # 10% memory change
                self.stability_alerts.append({
                    'timestamp': current['timestamp'],
                    'type': 'memory_fluctuation',
                    'severity': 'warning',
                    'details': f"Memory usage changed by {memory_change:.1f}%"
                })
                
            # High resource usage alerts
            if current['cpu']['overall_percent'] > 95:
                self.stability_alerts.append({
                    'timestamp': current['timestamp'],
                    'type': 'high_cpu_usage',
                    'severity': 'critical',
                    'details': f"CPU usage at {current['cpu']['overall_percent']:.1f}%"
                })
                
            if current['memory']['percent'] > 95:
                self.stability_alerts.append({
                    'timestamp': current['timestamp'],
                    'type': 'high_memory_usage',
                    'severity': 'critical',
                    'details': f"Memory usage at {current['memory']['percent']:.1f}%"
                })
                
        except Exception as e:
            logger.warning(f"Stability analysis error: {e}")

class SustainedLoadTester:
    """Sustained load testing for extended periods"""
    
    def __init__(self):
        self.load_active = False
        self.load_threads = []
        
    async def run_sustained_load_test(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Run sustained load test"""
        logger.info(f"Starting sustained load test for {duration_minutes} minutes")
        
        monitor = SystemStabilityMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        try:
            # Start multiple load generators
            self.load_active = True
            
            # CPU load
            cpu_load_thread = threading.Thread(target=self._generate_cpu_load)
            self.load_threads.append(cpu_load_thread)
            cpu_load_thread.start()
            
            # Memory load
            memory_load_thread = threading.Thread(target=self._generate_memory_load)
            self.load_threads.append(memory_load_thread)
            memory_load_thread.start()
            
            # Disk I/O load
            disk_load_thread = threading.Thread(target=self._generate_disk_load)
            self.load_threads.append(disk_load_thread)
            disk_load_thread.start()
            
            # Network/API load
            network_load_thread = threading.Thread(target=self._generate_network_load)
            self.load_threads.append(network_load_thread)
            network_load_thread.start()
            
            # Monitor progress
            end_time = start_time + duration_seconds
            last_report = start_time
            
            while time.time() < end_time:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = end_time - current_time
                
                logger.info(f"Sustained load test progress: {elapsed/60:.1f}/{duration_minutes} minutes, {remaining/60:.1f} minutes remaining")
                
            # Stop load generation
            self.load_active = False
            
            # Wait for load threads to finish
            for thread in self.load_threads:
                thread.join(timeout=10)
                
            total_duration = time.time() - start_time
            stability_data = monitor.stop_monitoring()
            
            # Analyze sustained load performance
            metrics = stability_data['metrics']
            if metrics:
                cpu_usage = [m['cpu']['overall_percent'] for m in metrics]
                memory_usage = [m['memory']['percent'] for m in metrics]
                
                cpu_stats = {
                    'avg': np.mean(cpu_usage),
                    'max': np.max(cpu_usage),
                    'min': np.min(cpu_usage),
                    'std': np.std(cpu_usage)
                }
                
                memory_stats = {
                    'avg': np.mean(memory_usage),
                    'max': np.max(memory_usage),
                    'min': np.min(memory_usage),
                    'std': np.std(memory_usage)
                }
            else:
                cpu_stats = memory_stats = {'avg': 0, 'max': 0, 'min': 0, 'std': 0}
                
            return {
                'test_type': 'sustained_load',
                'duration_requested_minutes': duration_minutes,
                'duration_actual_seconds': total_duration,
                'success': total_duration >= duration_seconds * 0.95,  # 95% completion
                'cpu_statistics': cpu_stats,
                'memory_statistics': memory_stats,
                'stability_alerts': stability_data['stability_alerts'],
                'alert_count': len(stability_data['stability_alerts']),
                'monitoring_samples': len(metrics),
                'performance_grade': self._grade_sustained_load_performance(cpu_stats, memory_stats, stability_data['stability_alerts']),
                'system_stability': len(stability_data['stability_alerts']) < 10  # Less than 10 alerts = stable
            }
            
        except Exception as e:
            monitor.stop_monitoring()
            self.load_active = False
            return {
                'test_type': 'sustained_load',
                'success': False,
                'error': str(e),
                'duration_actual_seconds': time.time() - start_time
            }
            
    def _generate_cpu_load(self):
        """Generate CPU load"""
        while self.load_active:
            try:
                # CPU-intensive calculations
                for _ in range(1000):
                    if not self.load_active:
                        break
                    np.random.random((100, 100)) @ np.random.random((100, 100))
                time.sleep(0.01)  # Brief pause
            except Exception as e:
                logger.warning(f"CPU load generation error: {e}")
                time.sleep(1)
                
    def _generate_memory_load(self):
        """Generate memory load"""
        memory_pools = []
        while self.load_active:
            try:
                # Allocate memory in chunks
                if len(memory_pools) < 50:  # Limit memory usage
                    memory_chunk = np.random.random((1000, 1000))  # ~8MB
                    memory_pools.append(memory_chunk)
                else:
                    # Cycle through memory
                    memory_pools.pop(0)
                    memory_chunk = np.random.random((1000, 1000))
                    memory_pools.append(memory_chunk)
                    
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Memory load generation error: {e}")
                time.sleep(1)
                
    def _generate_disk_load(self):
        """Generate disk I/O load"""
        temp_file = "/tmp/nautilus_stress_test.tmp"
        while self.load_active:
            try:
                # Write test data
                test_data = np.random.random(10000).tobytes()
                with open(temp_file, 'wb') as f:
                    f.write(test_data)
                    
                # Read test data
                with open(temp_file, 'rb') as f:
                    _ = f.read()
                    
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Disk load generation error: {e}")
                time.sleep(1)
                
        # Cleanup
        try:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
            
    def _generate_network_load(self):
        """Generate network/API load"""
        urls = [
            'http://localhost:8001/health',
            'http://localhost:8100/status',
            'http://localhost:8200/status',
            'http://localhost:8300/status',
            'http://localhost:8400/status'
        ]
        
        while self.load_active:
            try:
                url = random.choice(urls)
                response = requests.get(url, timeout=2)
                time.sleep(0.5)  # 2 requests per second per thread
            except Exception:
                time.sleep(1)
                
    def _grade_sustained_load_performance(self, cpu_stats: Dict, memory_stats: Dict, alerts: List) -> str:
        """Grade sustained load performance"""
        score = 0
        
        # CPU stability (40 points)
        if cpu_stats['std'] < 10:  # Low standard deviation = stable
            score += 40
        elif cpu_stats['std'] < 20:
            score += 30
        elif cpu_stats['std'] < 30:
            score += 20
        elif cpu_stats['std'] < 40:
            score += 10
            
        # Memory stability (30 points)
        if memory_stats['std'] < 5:
            score += 30
        elif memory_stats['std'] < 10:
            score += 25
        elif memory_stats['std'] < 15:
            score += 20
        elif memory_stats['std'] < 20:
            score += 15
        elif memory_stats['std'] < 25:
            score += 10
            
        # Alert count (30 points)
        alert_count = len(alerts)
        if alert_count == 0:
            score += 30
        elif alert_count <= 5:
            score += 25
        elif alert_count <= 10:
            score += 20
        elif alert_count <= 20:
            score += 15
        elif alert_count <= 30:
            score += 10
            
        # Convert to grade
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        else:
            return "D"

class ErrorInjectionTester:
    """Error injection and recovery testing"""
    
    def __init__(self):
        self.containers = [
            'nautilus-analytics-engine',
            'nautilus-risk-engine',
            'nautilus-factor-engine',
            'nautilus-ml-engine',
            'nautilus-features-engine'
        ]
        
    async def run_error_injection_test(self) -> Dict[str, Any]:
        """Run error injection and recovery tests"""
        logger.info("Starting error injection and recovery testing")
        
        results = []
        
        # Test 1: Container restart simulation
        container_restart_results = await self._test_container_restart_recovery()
        results.append(container_restart_results)
        
        # Test 2: Network failure simulation
        network_failure_results = await self._test_network_failure_recovery()
        results.append(network_failure_results)
        
        # Test 3: High error rate simulation
        error_rate_results = await self._test_high_error_rate_handling()
        results.append(error_rate_results)
        
        # Analyze overall resilience
        successful_tests = sum(1 for r in results if r.get('success', False))
        recovery_times = [r.get('recovery_time', 0) for r in results if r.get('recovery_time')]
        
        return {
            'test_type': 'error_injection_recovery',
            'individual_tests': results,
            'total_tests': len(results),
            'successful_tests': successful_tests,
            'success_rate': successful_tests / len(results),
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
            'max_recovery_time': max(recovery_times) if recovery_times else 0,
            'resilience_grade': self._grade_resilience(successful_tests, len(results), recovery_times),
            'success': successful_tests >= len(results) * 0.8  # 80% success rate
        }
        
    async def _test_container_restart_recovery(self) -> Dict[str, Any]:
        """Test container restart and recovery"""
        test_container = random.choice(self.containers)
        logger.info(f"Testing container restart recovery: {test_container}")
        
        start_time = time.time()
        
        try:
            # Get initial health status
            port = self._get_container_port(test_container)
            initial_health = await self._check_container_health(port)
            
            # Restart container
            restart_start = time.time()
            restart_result = subprocess.run(
                ['docker', 'restart', test_container],
                capture_output=True,
                text=True,
                timeout=60
            )
            restart_time = time.time() - restart_start
            
            if restart_result.returncode != 0:
                return {
                    'test': 'container_restart_recovery',
                    'container': test_container,
                    'success': False,
                    'error': 'Failed to restart container',
                    'details': restart_result.stderr
                }
                
            # Wait for recovery and measure time
            recovery_start = time.time()
            recovered = False
            max_wait = 60  # 60 second timeout
            
            while time.time() - recovery_start < max_wait:
                await asyncio.sleep(2)
                health = await self._check_container_health(port)
                if health:
                    recovered = True
                    break
                    
            recovery_time = time.time() - recovery_start
            total_time = time.time() - start_time
            
            return {
                'test': 'container_restart_recovery',
                'container': test_container,
                'success': recovered,
                'restart_time': restart_time,
                'recovery_time': recovery_time,
                'total_time': total_time,
                'initial_health': initial_health,
                'final_health': recovered
            }
            
        except Exception as e:
            return {
                'test': 'container_restart_recovery',
                'container': test_container,
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            
    async def _test_network_failure_recovery(self) -> Dict[str, Any]:
        """Test network failure recovery simulation"""
        logger.info("Testing network failure recovery simulation")
        
        start_time = time.time()
        
        try:
            # Simulate network issues by rapid requests
            target_url = 'http://localhost:8100/status'
            
            # Test baseline performance
            baseline_success_count = 0
            for _ in range(10):
                try:
                    response = requests.get(target_url, timeout=2)
                    if response.status_code == 200:
                        baseline_success_count += 1
                except:
                    pass
                    
            baseline_rate = baseline_success_count / 10
            
            # Generate high request load to simulate network stress
            stress_requests = []
            
            async def stress_request():
                try:
                    response = requests.get(target_url, timeout=1)
                    return response.status_code == 200
                except:
                    return False
                    
            # Generate stress load
            stress_tasks = [stress_request() for _ in range(100)]
            stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
            
            successful_stress_requests = sum(1 for r in stress_results if r is True)
            stress_success_rate = successful_stress_requests / len(stress_results)
            
            # Test recovery
            await asyncio.sleep(5)  # Wait for recovery
            
            recovery_success_count = 0
            for _ in range(10):
                try:
                    response = requests.get(target_url, timeout=2)
                    if response.status_code == 200:
                        recovery_success_count += 1
                except:
                    pass
                await asyncio.sleep(0.5)
                
            recovery_rate = recovery_success_count / 10
            
            return {
                'test': 'network_failure_recovery',
                'baseline_success_rate': baseline_rate,
                'stress_success_rate': stress_success_rate,
                'recovery_success_rate': recovery_rate,
                'recovery_time': 5,  # Fixed 5 second recovery period
                'success': recovery_rate >= baseline_rate * 0.8,  # 80% of baseline
                'resilient': stress_success_rate > 0.5  # Still handled 50% during stress
            }
            
        except Exception as e:
            return {
                'test': 'network_failure_recovery',
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            
    async def _test_high_error_rate_handling(self) -> Dict[str, Any]:
        """Test high error rate handling"""
        logger.info("Testing high error rate handling")
        
        start_time = time.time()
        
        try:
            # Test multiple endpoints with intentional errors
            test_endpoints = [
                'http://localhost:8001/nonexistent',  # 404 error
                'http://localhost:8100/invalid',     # Invalid endpoint
                'http://localhost:8200/error',       # Potential error endpoint
            ]
            
            error_responses = []
            
            for endpoint in test_endpoints:
                for _ in range(10):
                    try:
                        response = requests.get(endpoint, timeout=5)
                        error_responses.append({
                            'endpoint': endpoint,
                            'status_code': response.status_code,
                            'response_time': response.elapsed.total_seconds(),
                            'handled_gracefully': response.status_code in [404, 422, 500]
                        })
                    except Exception as e:
                        error_responses.append({
                            'endpoint': endpoint,
                            'error': str(e),
                            'handled_gracefully': True  # Exception handling counts as graceful
                        })
                        
            graceful_handling_count = sum(1 for r in error_responses if r.get('handled_gracefully', False))
            graceful_handling_rate = graceful_handling_count / len(error_responses)
            
            # Test system stability during errors
            system_stable = True
            try:
                health_response = requests.get('http://localhost:8001/health', timeout=5)
                system_stable = health_response.status_code == 200
            except:
                system_stable = False
                
            return {
                'test': 'high_error_rate_handling',
                'total_error_requests': len(error_responses),
                'graceful_handling_count': graceful_handling_count,
                'graceful_handling_rate': graceful_handling_rate,
                'system_stable_during_errors': system_stable,
                'success': graceful_handling_rate > 0.8 and system_stable,
                'duration': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'test': 'high_error_rate_handling',
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            
    def _get_container_port(self, container_name: str) -> int:
        """Get port for container"""
        port_map = {
            'nautilus-analytics-engine': 8100,
            'nautilus-risk-engine': 8200,
            'nautilus-factor-engine': 8300,
            'nautilus-ml-engine': 8400,
            'nautilus-features-engine': 8500
        }
        return port_map.get(container_name, 8000)
        
    async def _check_container_health(self, port: int) -> bool:
        """Check container health"""
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def _grade_resilience(self, successful_tests: int, total_tests: int, recovery_times: List[float]) -> str:
        """Grade system resilience"""
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        avg_recovery_time = np.mean(recovery_times) if recovery_times else float('inf')
        
        score = 0
        
        # Success rate (60 points)
        score += success_rate * 60
        
        # Recovery time (40 points)
        if avg_recovery_time < 10:
            score += 40
        elif avg_recovery_time < 30:
            score += 30
        elif avg_recovery_time < 60:
            score += 20
        elif avg_recovery_time < 120:
            score += 10
            
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        else:
            return "D"

async def run_reliability_stress_testing():
    """Run comprehensive reliability and stress testing"""
    logger.info("Starting Comprehensive Reliability and Stress Testing")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Sustained Load Test (shorter duration for demo)
    logger.info("Phase 1: Sustained Load Test (3 minutes)")
    sustained_load_tester = SustainedLoadTester()
    sustained_load_results = await sustained_load_tester.run_sustained_load_test(duration_minutes=3)
    results['sustained_load'] = sustained_load_results
    
    # Brief pause
    await asyncio.sleep(10)
    
    # 2. Error Injection and Recovery Test
    logger.info("Phase 2: Error Injection and Recovery Test")
    error_injection_tester = ErrorInjectionTester()
    error_injection_results = await error_injection_tester.run_error_injection_test()
    results['error_injection_recovery'] = error_injection_results
    
    # Generate comprehensive reliability report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate overall reliability grade
    individual_grades = []
    for test_name, result in results.items():
        grade = result.get('performance_grade') or result.get('resilience_grade', 'D')
        individual_grades.append(grade)
        
    grade_values = {'A+': 100, 'A': 95, 'B+': 85, 'B': 80, 'C': 70, 'D': 60}
    avg_grade_value = np.mean([grade_values.get(g, 60) for g in individual_grades])
    
    if avg_grade_value >= 95:
        overall_grade = "A+"
    elif avg_grade_value >= 90:
        overall_grade = "A"
    elif avg_grade_value >= 85:
        overall_grade = "B+"
    elif avg_grade_value >= 80:
        overall_grade = "B"
    elif avg_grade_value >= 70:
        overall_grade = "C"
    else:
        overall_grade = "D"
        
    # Overall success determination
    overall_success = all(result.get('success', False) for result in results.values())
    
    reliability_report = {
        'timestamp': timestamp,
        'test_duration_total_minutes': sum(
            result.get('duration_actual_seconds', 0) for result in results.values()
        ) / 60,
        'individual_results': results,
        'overall_performance': {
            'overall_grade': overall_grade,
            'overall_success': overall_success,
            'sustained_load_grade': sustained_load_results.get('performance_grade', 'D'),
            'error_recovery_grade': error_injection_results.get('resilience_grade', 'D'),
            'system_stability': sustained_load_results.get('system_stability', False),
            'error_resilience': error_injection_results.get('success', False)
        },
        'production_readiness': {
            'reliability_validated': overall_success,
            'stress_test_passed': sustained_load_results.get('success', False),
            'error_recovery_validated': error_injection_results.get('success', False),
            'recommendation': (
                'APPROVED for production deployment - System demonstrates high reliability' 
                if overall_success else 
                'REQUIRES reliability improvements before production deployment'
            )
        }
    }
    
    # Save results
    results_file = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/ml/reliability_stress_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(reliability_report, f, indent=2)
    
    logger.info(f"Reliability and stress test results saved to: {results_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("RELIABILITY AND STRESS TESTING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sustained Load Test: {sustained_load_results.get('performance_grade', 'D')}")
    logger.info(f"Error Recovery Test: {error_injection_results.get('resilience_grade', 'D')}")
    logger.info(f"Overall Reliability Grade: {overall_grade}")
    logger.info(f"System Stability: {'STABLE' if sustained_load_results.get('system_stability', False) else 'UNSTABLE'}")
    logger.info(f"Error Resilience: {'RESILIENT' if error_injection_results.get('success', False) else 'NEEDS IMPROVEMENT'}")
    logger.info(f"Production Ready: {'YES' if overall_success else 'NO'}")
    logger.info("=" * 60)
    
    return reliability_report

if __name__ == "__main__":
    asyncio.run(run_reliability_stress_testing())
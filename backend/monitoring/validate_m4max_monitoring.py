"""
M4 Max Monitoring System Validation Script
Comprehensive validation and testing of the M4 Max performance monitoring system.

Tests:
- M4 Max hardware detection and monitoring
- Container performance monitoring
- Trading performance monitoring
- Production dashboard functionality
- Prometheus metrics collection
- Grafana dashboard integration
- Alert system validation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import requests
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.m4max_hardware_monitor import M4MaxHardwareMonitor
from monitoring.container_performance_monitor import ContainerPerformanceMonitor
from monitoring.trading_performance_monitor import TradingPerformanceMonitor
from monitoring.production_monitoring_dashboard import ProductionMonitoringDashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class M4MaxMonitoringValidator:
    """Comprehensive validation system for M4 Max monitoring"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Test configuration
        self.test_duration = 30  # seconds
        self.backend_url = "http://localhost:8001"
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3002"
        
        logger.info("M4 Max Monitoring Validator initialized")
    
    def log_test_result(self, test_name: str, success: bool, details: str, metrics: Dict[str, Any] = None):
        """Log test result"""
        self.results['tests'][test_name] = {
            'success': success,
            'details': details,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}: {details}")
    
    async def test_m4max_hardware_monitoring(self) -> bool:
        """Test M4 Max hardware monitoring functionality"""
        test_name = "M4Max Hardware Monitoring"
        
        try:
            monitor = M4MaxHardwareMonitor()
            
            # Test single metrics collection
            metrics = monitor.collect_metrics()
            
            if not metrics:
                self.log_test_result(test_name, False, "Failed to collect M4 Max metrics")
                return False
            
            # Validate metrics structure
            required_fields = [
                'cpu_p_cores_usage', 'cpu_e_cores_usage', 'cpu_frequency_mhz',
                'unified_memory_usage_gb', 'unified_memory_bandwidth_gbps',
                'gpu_utilization_percent', 'neural_engine_utilization_percent',
                'neural_engine_tops_used', 'thermal_state', 'power_consumption_watts'
            ]
            
            missing_fields = []
            for field in required_fields:
                if not hasattr(metrics, field):
                    missing_fields.append(field)
            
            if missing_fields:
                self.log_test_result(test_name, False, f"Missing fields: {missing_fields}")
                return False
            
            # Test continuous monitoring for a short period
            monitor_task = asyncio.create_task(monitor.start_monitoring(interval=2.0))
            await asyncio.sleep(10)  # Monitor for 10 seconds
            monitor.stop_monitoring()
            monitor_task.cancel()
            
            # Test Prometheus metrics
            prometheus_metrics = monitor.get_prometheus_metrics()
            if 'm4max_cpu_p_cores_usage_percent' not in prometheus_metrics:
                self.log_test_result(test_name, False, "Prometheus metrics not generated correctly")
                return False
            
            # Test Redis storage
            current_metrics = monitor.get_current_metrics()
            if not current_metrics:
                self.log_test_result(test_name, False, "Failed to store/retrieve metrics from Redis")
                return False
            
            test_metrics = {
                'cpu_p_cores_usage': metrics.cpu_p_cores_usage,
                'cpu_e_cores_usage': metrics.cpu_e_cores_usage,
                'gpu_utilization': metrics.gpu_utilization_percent,
                'neural_engine_utilization': metrics.neural_engine_utilization_percent,
                'thermal_state': metrics.thermal_state,
                'is_m4_max_detected': monitor.is_m4_max
            }
            
            self.log_test_result(test_name, True, 
                               f"M4 Max monitoring successful (M4 Max detected: {monitor.is_m4_max})", 
                               test_metrics)
            
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_container_monitoring(self) -> bool:
        """Test container performance monitoring"""
        test_name = "Container Performance Monitoring"
        
        try:
            monitor = ContainerPerformanceMonitor()
            
            # Test container metrics collection
            container_metrics, engine_health = await monitor.collect_all_metrics()
            
            if not container_metrics:
                self.log_test_result(test_name, False, "No container metrics collected")
                return False
            
            # Validate container metrics structure
            nautilus_containers = [m for m in container_metrics if m.container_name.startswith('nautilus-')]
            
            if len(nautilus_containers) == 0:
                self.log_test_result(test_name, False, "No Nautilus containers detected")
                return False
            
            # Test engine health checks
            if not engine_health:
                self.log_test_result(test_name, False, "No engine health metrics collected")
                return False
            
            # Test Prometheus metrics
            prometheus_metrics = monitor.get_prometheus_metrics()
            if 'nautilus_container_cpu_usage_percent' not in prometheus_metrics:
                self.log_test_result(test_name, False, "Container Prometheus metrics not generated")
                return False
            
            # Test container status summary
            status_summary = monitor.get_container_status_summary()
            if 'total_containers' not in status_summary:
                self.log_test_result(test_name, False, "Container status summary not available")
                return False
            
            test_metrics = {
                'total_containers': len(container_metrics),
                'nautilus_containers': len(nautilus_containers),
                'running_containers': status_summary.get('running', 0),
                'engines_monitored': len(engine_health),
                'healthy_engines': sum(1 for h in engine_health if h.is_healthy)
            }
            
            self.log_test_result(test_name, True, 
                               f"Container monitoring successful ({len(nautilus_containers)} Nautilus containers)", 
                               test_metrics)
            
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_trading_monitoring(self) -> bool:
        """Test trading performance monitoring"""
        test_name = "Trading Performance Monitoring"
        
        try:
            monitor = TradingPerformanceMonitor()
            
            # Test trading metrics collection
            metrics = await monitor.collect_all_trading_metrics()
            
            if 'timestamp' not in metrics:
                self.log_test_result(test_name, False, "Trading metrics collection failed")
                return False
            
            # Validate metrics structure
            required_sections = ['latency_metrics', 'risk_metrics', 'ml_metrics', 'throughput_metrics']
            missing_sections = [section for section in required_sections if section not in metrics]
            
            if missing_sections:
                self.log_test_result(test_name, False, f"Missing sections: {missing_sections}")
                return False
            
            # Test performance score calculation
            overall_score = metrics.get('overall_score', 0)
            if not isinstance(overall_score, (int, float)) or overall_score < 0 or overall_score > 100:
                self.log_test_result(test_name, False, f"Invalid overall score: {overall_score}")
                return False
            
            # Test Prometheus metrics
            prometheus_metrics = monitor.get_prometheus_metrics()
            if 'nautilus_overall_performance_score' not in prometheus_metrics:
                self.log_test_result(test_name, False, "Trading Prometheus metrics not generated")
                return False
            
            # Test performance summary
            summary = monitor.get_performance_summary()
            if 'error' in summary:
                self.log_test_result(test_name, False, f"Performance summary error: {summary['error']}")
                return False
            
            test_metrics = {
                'overall_score': overall_score,
                'latency_metrics_count': len(metrics.get('latency_metrics', [])),
                'ml_metrics_count': len(metrics.get('ml_metrics', [])),
                'throughput_metrics_count': len(metrics.get('throughput_metrics', [])),
                'has_risk_metrics': bool(metrics.get('risk_metrics'))
            }
            
            self.log_test_result(test_name, True, 
                               f"Trading monitoring successful (score: {overall_score:.1f})", 
                               test_metrics)
            
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_production_dashboard(self) -> bool:
        """Test production monitoring dashboard"""
        test_name = "Production Monitoring Dashboard"
        
        try:
            dashboard = ProductionMonitoringDashboard()
            
            # Test comprehensive metrics collection
            all_metrics = await dashboard.collect_all_metrics()
            
            if 'timestamp' not in all_metrics:
                self.log_test_result(test_name, False, "Dashboard metrics collection failed")
                return False
            
            # Validate comprehensive metrics structure
            required_sections = ['m4max_metrics', 'container_metrics', 'trading_metrics', 'system_health']
            missing_sections = [section for section in required_sections if section not in all_metrics]
            
            if missing_sections:
                self.log_test_result(test_name, False, f"Missing sections: {missing_sections}")
                return False
            
            # Test system health calculation
            system_health = all_metrics.get('system_health', {})
            if 'overall_status' not in system_health:
                self.log_test_result(test_name, False, "System health calculation failed")
                return False
            
            # Test alert evaluation
            active_alerts = all_metrics.get('active_alerts', [])
            alert_count = len(active_alerts)
            
            # Test dashboard summary
            summary = dashboard.get_dashboard_summary()
            if 'error' in summary:
                self.log_test_result(test_name, False, f"Dashboard summary error: {summary['error']}")
                return False
            
            # Test Prometheus metrics compilation
            prometheus_metrics = dashboard.get_prometheus_metrics()
            if not prometheus_metrics:
                self.log_test_result(test_name, False, "Dashboard Prometheus metrics compilation failed")
                return False
            
            test_metrics = {
                'system_status': system_health.get('overall_status'),
                'performance_score': system_health.get('performance_score', 0),
                'active_alerts': alert_count,
                'uptime_seconds': system_health.get('uptime_seconds', 0),
                'optimization_opportunities': len(system_health.get('optimization_opportunities', [])),
                'recommendations': len(system_health.get('recommendations', []))
            }
            
            self.log_test_result(test_name, True, 
                               f"Dashboard operational (status: {system_health.get('overall_status')})", 
                               test_metrics)
            
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test monitoring API endpoints"""
        test_name = "API Endpoints"
        
        try:
            # Test endpoints
            endpoints_to_test = [
                "/api/v1/monitoring/health",
                "/api/v1/monitoring/system/health",
                "/api/v1/monitoring/status",
                "/api/v1/monitoring/metrics"
            ]
            
            successful_endpoints = 0
            endpoint_results = {}
            
            for endpoint in endpoints_to_test:
                try:
                    url = f"{self.backend_url}{endpoint}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        successful_endpoints += 1
                        endpoint_results[endpoint] = "success"
                    else:
                        endpoint_results[endpoint] = f"status_code_{response.status_code}"
                        
                except requests.exceptions.RequestException as e:
                    endpoint_results[endpoint] = f"connection_error: {str(e)[:50]}"
            
            if successful_endpoints == 0:
                self.log_test_result(test_name, False, "No API endpoints accessible")
                return False
            
            success_rate = (successful_endpoints / len(endpoints_to_test)) * 100
            
            test_metrics = {
                'successful_endpoints': successful_endpoints,
                'total_endpoints': len(endpoints_to_test),
                'success_rate': success_rate,
                'endpoint_results': endpoint_results
            }
            
            if success_rate >= 50:  # At least 50% of endpoints working
                self.log_test_result(test_name, True, 
                                   f"API endpoints accessible ({successful_endpoints}/{len(endpoints_to_test)})", 
                                   test_metrics)
                return True
            else:
                self.log_test_result(test_name, False, 
                                   f"Too many API endpoints failing ({successful_endpoints}/{len(endpoints_to_test)})", 
                                   test_metrics)
                return False
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_prometheus_integration(self) -> bool:
        """Test Prometheus integration"""
        test_name = "Prometheus Integration"
        
        try:
            # Test Prometheus availability
            try:
                response = requests.get(f"{self.prometheus_url}/api/v1/label/__name__/values", timeout=5)
                if response.status_code != 200:
                    self.log_test_result(test_name, False, f"Prometheus not accessible: {response.status_code}")
                    return False
            except requests.exceptions.RequestException:
                self.log_test_result(test_name, False, "Prometheus connection failed")
                return False
            
            # Check for M4 Max specific metrics
            m4max_metrics_to_check = [
                'm4max_cpu_p_cores_usage_percent',
                'm4max_unified_memory_usage_gb',
                'm4max_gpu_utilization_percent',
                'm4max_neural_engine_utilization_percent'
            ]
            
            found_metrics = []
            try:
                metrics_response = requests.get(f"{self.prometheus_url}/api/v1/label/__name__/values", timeout=5)
                if metrics_response.status_code == 200:
                    all_metrics = metrics_response.json().get('data', [])
                    found_metrics = [metric for metric in m4max_metrics_to_check if metric in all_metrics]
            except:
                pass  # Continue with partial test
            
            # Test Prometheus query
            try:
                query_response = requests.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={'query': 'up'},
                    timeout=5
                )
                query_successful = query_response.status_code == 200
            except:
                query_successful = False
            
            test_metrics = {
                'prometheus_accessible': True,
                'query_functional': query_successful,
                'm4max_metrics_found': len(found_metrics),
                'total_m4max_metrics_expected': len(m4max_metrics_to_check),
                'found_metrics': found_metrics
            }
            
            if query_successful:
                self.log_test_result(test_name, True, 
                                   f"Prometheus integration working ({len(found_metrics)} M4 Max metrics found)", 
                                   test_metrics)
                return True
            else:
                self.log_test_result(test_name, False, "Prometheus queries not working", test_metrics)
                return False
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def test_grafana_integration(self) -> bool:
        """Test Grafana integration"""
        test_name = "Grafana Integration"
        
        try:
            # Test Grafana availability
            try:
                response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
                if response.status_code != 200:
                    self.log_test_result(test_name, False, f"Grafana not accessible: {response.status_code}")
                    return False
            except requests.exceptions.RequestException:
                self.log_test_result(test_name, False, "Grafana connection failed")
                return False
            
            # Test dashboard availability (if possible)
            dashboard_accessible = False
            try:
                # Try to access dashboards list
                dashboards_response = requests.get(f"{self.grafana_url}/api/search", timeout=5)
                if dashboards_response.status_code == 200:
                    dashboard_accessible = True
            except:
                pass  # Continue with basic test
            
            test_metrics = {
                'grafana_accessible': True,
                'dashboard_api_accessible': dashboard_accessible
            }
            
            self.log_test_result(test_name, True, "Grafana integration accessible", test_metrics)
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of M4 Max monitoring system"""
        logger.info("🚀 Starting comprehensive M4 Max monitoring system validation...")
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_m4max_hardware_monitoring(),
            self.test_container_monitoring(),
            self.test_trading_monitoring(),
            self.test_production_dashboard(),
            self.test_api_endpoints(),
            self.test_prometheus_integration(),
            self.test_grafana_integration()
        ]
        
        # Execute tests concurrently where possible
        test_results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Calculate overall results
        successful_tests = sum(1 for result in test_results if result is True)
        total_tests = len(test_results)
        success_rate = (successful_tests / total_tests) * 100
        
        # Determine overall status
        if success_rate >= 90:
            overall_status = "excellent"
        elif success_rate >= 70:
            overall_status = "good"
        elif success_rate >= 50:
            overall_status = "acceptable"
        else:
            overall_status = "poor"
        
        # Generate recommendations
        recommendations = []
        
        if not self.results['tests'].get('M4Max Hardware Monitoring', {}).get('success', False):
            recommendations.append("M4 Max hardware monitoring needs attention - check system compatibility")
        
        if not self.results['tests'].get('Container Performance Monitoring', {}).get('success', False):
            recommendations.append("Container monitoring issues - verify Docker containers are running")
        
        if not self.results['tests'].get('Trading Performance Monitoring', {}).get('success', False):
            recommendations.append("Trading performance monitoring needs configuration")
        
        if not self.results['tests'].get('API Endpoints', {}).get('success', False):
            recommendations.append("API endpoints not accessible - check backend service")
        
        if not self.results['tests'].get('Prometheus Integration', {}).get('success', False):
            recommendations.append("Prometheus integration issues - verify Prometheus is running and configured")
        
        if not self.results['tests'].get('Grafana Integration', {}).get('success', False):
            recommendations.append("Grafana integration issues - verify Grafana is running and accessible")
        
        # Finalize results
        self.results.update({
            'overall_status': overall_status,
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'validation_duration_seconds': time.time() - start_time,
            'recommendations': recommendations
        })
        
        return self.results
    
    def print_validation_report(self):
        """Print detailed validation report"""
        print("\n" + "="*80)
        print("🎯 M4 MAX MONITORING SYSTEM VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Status: {self.results['overall_status'].upper()}")
        print(f"✅ Success Rate: {self.results['success_rate']:.1f}% ({self.results['successful_tests']}/{self.results['total_tests']})")
        print(f"⏱️  Validation Duration: {self.results['validation_duration_seconds']:.2f} seconds")
        
        print(f"\n🧪 Test Results:")
        print("-" * 50)
        
        for test_name, test_result in self.results['tests'].items():
            status_icon = "✅" if test_result['success'] else "❌"
            print(f"{status_icon} {test_name}")
            print(f"   Details: {test_result['details']}")
            
            if test_result.get('metrics'):
                print("   Key Metrics:")
                for key, value in test_result['metrics'].items():
                    print(f"     - {key}: {value}")
            print()
        
        if self.results['recommendations']:
            print("💡 Recommendations:")
            print("-" * 30)
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\n📅 Validation completed at: {self.results['timestamp']}")
        print("="*80)

async def main():
    """Main validation execution"""
    validator = M4MaxMonitoringValidator()
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Print detailed report
        validator.print_validation_report()
        
        # Save results to file
        results_file = f"m4max_monitoring_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Validation results saved to: {results_file}")
        
        # Exit with appropriate code
        if results['success_rate'] >= 70:
            print("\n🎉 M4 Max monitoring system validation completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  M4 Max monitoring system needs attention before production use.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
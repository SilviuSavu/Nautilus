#!/usr/bin/env python3
"""
CPU Optimization Deployment Test Script
=======================================

Tests the CPU optimization system deployment on M4 Max architecture.
Validates core detection, container integration, and performance monitoring.
"""

import os
import sys
import time
import json
import asyncio
import logging
import requests
import subprocess
from typing import Dict, List, Any
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.optimizer_controller import OptimizerController, OptimizationMode
from optimization.cpu_affinity import WorkloadPriority
from optimization.process_manager import ProcessClass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CPUOptimizationTester:
    """Test suite for CPU optimization deployment"""
    
    def __init__(self):
        self.optimizer_controller: OptimizerController = None
        self.backend_url = "http://localhost:8001"
        self.test_results = {
            "timestamp": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": [],
            "system_info": {},
            "performance_metrics": {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive CPU optimization tests"""
        print("🚀 Starting CPU Optimization Deployment Tests for M4 Max...")
        
        try:
            # Test 1: Core Detection and Architecture
            self.test_core_detection()
            
            # Test 2: Optimizer Controller Initialization
            self.test_optimizer_initialization()
            
            # Test 3: Process Registration and Management
            self.test_process_management()
            
            # Test 4: Container Integration
            self.test_container_integration()
            
            # Test 5: Performance Monitoring
            self.test_performance_monitoring()
            
            # Test 6: API Endpoints
            self.test_api_endpoints()
            
            # Test 7: Latency Measurement
            self.test_latency_measurement()
            
            # Test 8: Workload Classification
            self.test_workload_classification()
            
            # Test 9: GCD Integration (macOS only)
            if sys.platform == "darwin":
                self.test_gcd_integration()
            
            # Test 10: Load Testing and Optimization
            self.test_optimization_under_load()
            
        except Exception as e:
            self.record_test_failure("Critical", f"Test suite failure: {e}")
            
        finally:
            # Cleanup
            if self.optimizer_controller:
                self.optimizer_controller.shutdown()
        
        return self.generate_test_report()
    
    def test_core_detection(self):
        """Test M4 Max core detection and configuration"""
        print("\n🔍 Test 1: M4 Max Core Detection")
        
        try:
            cpu_count = os.cpu_count()
            
            if cpu_count == 16:
                self.record_test_success("Core Detection", 
                                       f"Detected M4 Max configuration: {cpu_count} cores")
                
                # Validate P-core and E-core configuration
                expected_p_cores = list(range(12))
                expected_e_cores = list(range(12, 16))
                
                self.test_results["system_info"]["total_cores"] = cpu_count
                self.test_results["system_info"]["p_cores"] = expected_p_cores
                self.test_results["system_info"]["e_cores"] = expected_e_cores
                self.test_results["system_info"]["architecture"] = "M4 Max"
                
            else:
                self.record_test_warning("Core Detection", 
                                       f"Non-M4 Max configuration detected: {cpu_count} cores")
                self.test_results["system_info"]["architecture"] = "Generic"
                
        except Exception as e:
            self.record_test_failure("Core Detection", f"Core detection failed: {e}")
    
    def test_optimizer_initialization(self):
        """Test optimizer controller initialization"""
        print("\n⚡ Test 2: Optimizer Controller Initialization")
        
        try:
            start_time = time.time()
            self.optimizer_controller = OptimizerController()
            
            success = self.optimizer_controller.initialize()
            init_time = time.time() - start_time
            
            if success:
                self.record_test_success("Initialization", 
                                       f"Optimizer initialized in {init_time:.2f}s")
                
                # Get system information
                system_info = self.optimizer_controller.get_comprehensive_stats()
                self.test_results["system_info"].update(system_info.get("system_info", {}))
                
            else:
                self.record_test_failure("Initialization", "Failed to initialize optimizer")
                
        except Exception as e:
            self.record_test_failure("Initialization", f"Initialization error: {e}")
    
    def test_process_management(self):
        """Test process registration and management"""
        print("\n👥 Test 3: Process Management")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("Process Management", "Optimizer not initialized")
                return
            
            # Register current process for testing
            current_pid = os.getpid()
            
            success = self.optimizer_controller.register_process(
                current_pid,
                ProcessClass.ANALYTICS,
                WorkloadPriority.NORMAL
            )
            
            if success:
                self.record_test_success("Process Management", 
                                       f"Registered PID {current_pid} successfully")
                
                # Get process stats
                process_stats = self.optimizer_controller.process_manager.get_process_stats()
                self.test_results["performance_metrics"]["managed_processes"] = process_stats
                
            else:
                self.record_test_failure("Process Management", "Failed to register process")
                
        except Exception as e:
            self.record_test_failure("Process Management", f"Process management error: {e}")
    
    def test_container_integration(self):
        """Test container CPU optimization"""
        print("\n🐳 Test 4: Container Integration")
        
        try:
            if not self.optimizer_controller or not self.optimizer_controller.container_optimizer:
                self.record_test_warning("Container Integration", "Container optimizer not available")
                return
            
            # Get container stats
            container_stats = self.optimizer_controller.container_optimizer.get_container_stats()
            
            if container_stats["total_containers"] > 0:
                self.record_test_success("Container Integration", 
                                       f"Managing {container_stats['total_containers']} containers")
                
                # Force container optimization
                result = self.optimizer_controller.container_optimizer.force_container_optimization()
                
                if result["success"]:
                    self.record_test_success("Container Optimization", 
                                           f"Optimized {result['containers_optimized']} containers")
                else:
                    self.record_test_warning("Container Optimization", "No containers optimized")
                
                self.test_results["performance_metrics"]["container_stats"] = container_stats
                
            else:
                self.record_test_warning("Container Integration", "No containers found to manage")
                
        except Exception as e:
            self.record_test_failure("Container Integration", f"Container integration error: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring system"""
        print("\n📊 Test 5: Performance Monitoring")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("Performance Monitoring", "Optimizer not initialized")
                return
            
            # Get system health
            health = self.optimizer_controller.get_system_health()
            
            self.record_test_success("Performance Monitoring", 
                                   f"System health: CPU {health.cpu_utilization:.1f}%, "
                                   f"Memory {health.memory_utilization:.1f}%, "
                                   f"Optimization Score {health.optimization_score:.2f}")
            
            # Get performance monitor stats
            perf_stats = self.optimizer_controller.performance_monitor.get_system_stats()
            self.test_results["performance_metrics"]["monitoring"] = perf_stats
            
        except Exception as e:
            self.record_test_failure("Performance Monitoring", f"Performance monitoring error: {e}")
    
    def test_api_endpoints(self):
        """Test optimization API endpoints"""
        print("\n🌐 Test 6: API Endpoints")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.backend_url}/api/v1/optimization/ping", timeout=5)
            
            if response.status_code == 200:
                self.record_test_success("API Health", "Optimization API is responding")
            else:
                self.record_test_failure("API Health", f"API health check failed: {response.status_code}")
                return
            
            # Test system info endpoint
            try:
                response = requests.get(f"{self.backend_url}/api/v1/optimization/system-info", timeout=10)
                if response.status_code == 200:
                    system_info = response.json()
                    self.record_test_success("API System Info", 
                                           f"Retrieved system info: {system_info.get('architecture', 'Unknown')}")
                else:
                    self.record_test_warning("API System Info", f"System info endpoint failed: {response.status_code}")
            except Exception:
                self.record_test_warning("API System Info", "System info endpoint not available")
            
            # Test container stats endpoint (if available)
            try:
                response = requests.get(f"{self.backend_url}/api/v1/optimization/containers/stats", timeout=10)
                if response.status_code == 200:
                    container_stats = response.json()
                    self.record_test_success("API Container Stats", 
                                           f"Retrieved container stats: {container_stats.get('total_containers', 0)} containers")
                elif response.status_code == 503:
                    self.record_test_warning("API Container Stats", "Container optimizer not available via API")
                else:
                    self.record_test_warning("API Container Stats", f"Container stats endpoint failed: {response.status_code}")
            except Exception:
                self.record_test_warning("API Container Stats", "Container stats endpoint not available")
            
        except requests.exceptions.ConnectionError:
            self.record_test_warning("API Endpoints", "Backend not running - skipping API tests")
        except Exception as e:
            self.record_test_failure("API Endpoints", f"API test error: {e}")
    
    def test_latency_measurement(self):
        """Test latency measurement functionality"""
        print("\n⏱️  Test 7: Latency Measurement")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("Latency Measurement", "Optimizer not initialized")
                return
            
            # Start latency measurement
            operation_id = self.optimizer_controller.start_latency_measurement("test_operation")
            
            if operation_id:
                # Simulate some work
                time.sleep(0.01)  # 10ms
                
                # End measurement
                latency_ms = self.optimizer_controller.end_latency_measurement(operation_id)
                
                self.record_test_success("Latency Measurement", 
                                       f"Measured latency: {latency_ms:.2f}ms for operation {operation_id}")
                
                # Get latency stats
                stats = self.optimizer_controller.performance_monitor.get_latency_stats("test_operation")
                self.test_results["performance_metrics"]["latency_stats"] = stats
                
            else:
                self.record_test_failure("Latency Measurement", "Failed to start latency measurement")
                
        except Exception as e:
            self.record_test_failure("Latency Measurement", f"Latency measurement error: {e}")
    
    def test_workload_classification(self):
        """Test workload classification system"""
        print("\n🤖 Test 8: Workload Classification")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("Workload Classification", "Optimizer not initialized")
                return
            
            # Test workload classification
            test_functions = [
                ("execute_order", "trading_engine"),
                ("process_market_data", "market_data"),
                ("calculate_risk", "risk_management"),
                ("generate_report", "analytics"),
                ("cleanup_data", "background")
            ]
            
            classification_results = []
            
            for func_name, module_name in test_functions:
                try:
                    category, priority = self.optimizer_controller.classify_and_optimize_workload(
                        func_name, module_name
                    )
                    
                    classification_results.append({
                        "function": func_name,
                        "module": module_name,
                        "category": category.value,
                        "priority": priority.name
                    })
                    
                except Exception as e:
                    self.record_test_warning("Workload Classification", 
                                           f"Classification failed for {func_name}: {e}")
            
            if classification_results:
                self.record_test_success("Workload Classification", 
                                       f"Classified {len(classification_results)} workloads")
                self.test_results["performance_metrics"]["classifications"] = classification_results
            else:
                self.record_test_failure("Workload Classification", "No workloads classified")
                
        except Exception as e:
            self.record_test_failure("Workload Classification", f"Workload classification error: {e}")
    
    def test_gcd_integration(self):
        """Test Grand Central Dispatch integration (macOS only)"""
        print("\n🍎 Test 9: GCD Integration (macOS)")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("GCD Integration", "Optimizer not initialized")
                return
            
            # Test GCD task dispatch
            def test_task():
                return "GCD task completed"
            
            task_id = self.optimizer_controller.dispatch_task(
                "trading.analytics",
                test_task
            )
            
            if task_id:
                self.record_test_success("GCD Integration", 
                                       f"Dispatched task {task_id} to GCD scheduler")
                
                # Get GCD stats
                gcd_stats = self.optimizer_controller.gcd_scheduler.get_system_stats()
                self.test_results["performance_metrics"]["gcd_stats"] = gcd_stats
                
            else:
                self.record_test_failure("GCD Integration", "Failed to dispatch GCD task")
                
        except Exception as e:
            self.record_test_failure("GCD Integration", f"GCD integration error: {e}")
    
    def test_optimization_under_load(self):
        """Test optimization system under simulated load"""
        print("\n🏋️ Test 10: Optimization Under Load")
        
        try:
            if not self.optimizer_controller:
                self.record_test_failure("Load Testing", "Optimizer not initialized")
                return
            
            # Get initial metrics
            initial_health = self.optimizer_controller.get_system_health()
            
            # Simulate CPU load
            def cpu_intensive_task():
                start = time.time()
                while (time.time() - start) < 2:  # Run for 2 seconds
                    _ = sum(i * i for i in range(1000))
                return "Load test completed"
            
            # Dispatch multiple tasks
            task_ids = []
            for i in range(4):  # 4 concurrent tasks
                task_id = self.optimizer_controller.dispatch_task(
                    "trading.analytics",
                    cpu_intensive_task
                )
                if task_id:
                    task_ids.append(task_id)
            
            if task_ids:
                self.record_test_success("Load Generation", 
                                       f"Started {len(task_ids)} load test tasks")
                
                # Wait for tasks and check system response
                time.sleep(3)
                
                # Get final metrics
                final_health = self.optimizer_controller.get_system_health()
                
                # Check if system maintained stability
                cpu_increase = final_health.cpu_utilization - initial_health.cpu_utilization
                
                if final_health.optimization_score >= 0.3:  # Reasonable threshold
                    self.record_test_success("Load Response", 
                                           f"System maintained stability under load "
                                           f"(CPU +{cpu_increase:.1f}%, Score: {final_health.optimization_score:.2f})")
                else:
                    self.record_test_warning("Load Response", 
                                           f"System performance degraded under load "
                                           f"(Score: {final_health.optimization_score:.2f})")
                
                self.test_results["performance_metrics"]["load_test"] = {
                    "initial_health": initial_health.__dict__,
                    "final_health": final_health.__dict__,
                    "tasks_completed": len(task_ids)
                }
                
            else:
                self.record_test_failure("Load Testing", "Failed to generate load")
                
        except Exception as e:
            self.record_test_failure("Load Testing", f"Load testing error: {e}")
    
    def record_test_success(self, test_name: str, message: str):
        """Record a successful test"""
        print(f"  ✅ {test_name}: {message}")
        self.test_results["tests_passed"] += 1
        self.test_results["test_details"].append({
            "test": test_name,
            "status": "PASS",
            "message": message,
            "timestamp": time.time()
        })
    
    def record_test_failure(self, test_name: str, message: str):
        """Record a failed test"""
        print(f"  ❌ {test_name}: {message}")
        self.test_results["tests_failed"] += 1
        self.test_results["test_details"].append({
            "test": test_name,
            "status": "FAIL",
            "message": message,
            "timestamp": time.time()
        })
    
    def record_test_warning(self, test_name: str, message: str):
        """Record a test warning"""
        print(f"  ⚠️  {test_name}: {message}")
        self.test_results["test_details"].append({
            "test": test_name,
            "status": "WARN",
            "message": message,
            "timestamp": time.time()
        })
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = self.test_results["tests_passed"] + self.test_results["tests_failed"]
        success_rate = (self.test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 CPU Optimization Deployment Test Results")
        print(f"="*60)
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Architecture: {self.test_results['system_info'].get('architecture', 'Unknown')}")
        
        if success_rate >= 80:
            print(f"🎉 DEPLOYMENT SUCCESS: CPU optimization system ready for production!")
        elif success_rate >= 60:
            print(f"⚠️  DEPLOYMENT WARNING: Some issues detected, review recommended")
        else:
            print(f"❌ DEPLOYMENT FAILED: Critical issues must be resolved")
        
        # Add summary to results
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "success_rate": success_rate,
            "deployment_status": "SUCCESS" if success_rate >= 80 else "WARNING" if success_rate >= 60 else "FAILED",
            "recommendations": self._generate_recommendations()
        }
        
        return self.test_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations based on test results"""
        recommendations = []
        
        failed_tests = [t for t in self.test_results["test_details"] if t["status"] == "FAIL"]
        
        if any("Initialization" in t["test"] for t in failed_tests):
            recommendations.append("Check system dependencies and permissions for optimizer initialization")
        
        if any("Container" in t["test"] for t in failed_tests):
            recommendations.append("Verify Docker daemon is running and accessible")
        
        if any("API" in t["test"] for t in failed_tests):
            recommendations.append("Ensure backend server is running on localhost:8001")
        
        if self.test_results["system_info"].get("architecture") != "M4 Max":
            recommendations.append("System optimized for M4 Max - performance may vary on other architectures")
        
        if not recommendations:
            recommendations.append("All tests passed - system ready for production deployment")
        
        return recommendations

def main():
    """Main test execution function"""
    tester = CPUOptimizationTester()
    results = tester.run_all_tests()
    
    # Save results to file
    timestamp = int(time.time())
    results_file = f"cpu_optimization_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Test results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    if results["summary"]["deployment_status"] == "SUCCESS":
        sys.exit(0)
    elif results["summary"]["deployment_status"] == "WARNING":
        sys.exit(1)
    else:
        sys.exit(2)
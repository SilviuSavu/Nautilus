#!/usr/bin/env python3
"""
Comprehensive System Performance Test Suite
Orchestrates all individual engine tests and integration tests to provide complete system analysis
Generates comprehensive performance reports with recommendations and M4 Max optimization insights
"""
import asyncio
import json
import logging
import time
import statistics
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import psutil
import docker
from pathlib import Path

# Import all individual test modules
from analytics_engine_performance_test import AnalyticsEnginePerformanceTest
from risk_engine_performance_test import RiskEnginePerformanceTest  
from factor_engine_performance_test import FactorEnginePerformanceTest
from ml_engine_performance_test import MLEnginePerformanceTest
from remaining_engines_performance_test import RemainingEnginesPerformanceTest
from inter_engine_integration_test import InterEngineIntegrationTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemPerformanceTest:
    """
    Master test suite that orchestrates all individual engine tests and integration tests
    Provides complete system performance analysis with M4 Max optimization insights
    """
    
    def __init__(self):
        self.test_start_time = datetime.now()
        self.docker_client = docker.from_env()
        self.system_info = self.collect_system_info()
        self.test_results = {}
        self.performance_summary = {}
        self.recommendations = []
        
        # Test instances
        self.test_instances = {
            "analytics_engine": AnalyticsEnginePerformanceTest(),
            "risk_engine": RiskEnginePerformanceTest(),
            "factor_engine": FactorEnginePerformanceTest(),
            "ml_engine": MLEnginePerformanceTest(),
            "remaining_engines": RemainingEnginesPerformanceTest(),
            "integration_tests": InterEngineIntegrationTest()
        }
        
        # M4 Max optimization tracking
        self.m4_max_metrics = self.detect_m4_max_capabilities()
        
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        logger.info("Collecting system information...")
        
        try:
            # Basic system info
            system_info = {
                "timestamp": datetime.now().isoformat(),
                "cpu_count": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "platform": "macOS" if "darwin" in psutil.LINUX else "unknown"
            }
            
            # Docker containers status
            try:
                containers = self.docker_client.containers.list(all=True)
                nautilus_containers = [c for c in containers if "nautilus" in c.name.lower()]
                
                system_info["docker_status"] = {
                    "total_containers": len(containers),
                    "nautilus_containers": len(nautilus_containers),
                    "running_containers": len([c for c in nautilus_containers if c.status == "running"]),
                    "container_details": [
                        {
                            "name": c.name,
                            "status": c.status,
                            "image": c.image.tags[0] if c.image.tags else "unknown"
                        }
                        for c in nautilus_containers
                    ]
                }
            except Exception as e:
                system_info["docker_status"] = {"error": str(e)}
            
            # Network connectivity check
            system_info["network_status"] = self.check_network_connectivity()
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error collecting system info: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def detect_m4_max_capabilities(self) -> Dict[str, Any]:
        """Detect M4 Max specific capabilities and optimization status"""
        logger.info("Detecting M4 Max capabilities...")
        
        m4_max_info = {
            "detected": False,
            "metal_gpu_available": False,
            "neural_engine_available": False,
            "unified_memory_detected": False,
            "optimization_status": "unknown"
        }
        
        try:
            # Check if running on Apple Silicon
            result = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
            if result.returncode == 0 and "Mac" in result.stdout:
                m4_max_info["detected"] = True
                m4_max_info["model"] = result.stdout.strip()
            
            # Check for Metal GPU support (Python)
            try:
                result = subprocess.run(["python3", "-c", "import torch; print(torch.backends.mps.is_available())"], 
                                      capture_output=True, text=True, timeout=10)
                if "True" in result.stdout:
                    m4_max_info["metal_gpu_available"] = True
            except:
                pass
            
            # Check memory configuration
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            if result.returncode == 0:
                memory_bytes = int(result.stdout.strip())
                memory_gb = memory_bytes / (1024**3)
                m4_max_info["total_memory_gb"] = memory_gb
                # M4 Max typically has 36GB+ unified memory
                if memory_gb >= 32:
                    m4_max_info["unified_memory_detected"] = True
            
            # Check CPU core configuration
            try:
                # Performance cores
                result_p = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True)
                # Efficiency cores  
                result_e = subprocess.run(["sysctl", "-n", "hw.perflevel1.logicalcpu"], capture_output=True, text=True)
                
                if result_p.returncode == 0 and result_e.returncode == 0:
                    p_cores = int(result_p.stdout.strip())
                    e_cores = int(result_e.stdout.strip())
                    m4_max_info["cpu_config"] = {
                        "performance_cores": p_cores,
                        "efficiency_cores": e_cores,
                        "total_cores": p_cores + e_cores
                    }
                    # M4 Max has 12P + 4E cores
                    if p_cores >= 10 and e_cores >= 4:
                        m4_max_info["m4_max_cpu_detected"] = True
            except:
                pass
            
            # Determine optimization status
            if (m4_max_info["detected"] and 
                m4_max_info["metal_gpu_available"] and 
                m4_max_info["unified_memory_detected"]):
                m4_max_info["optimization_status"] = "available"
            elif m4_max_info["detected"]:
                m4_max_info["optimization_status"] = "partial"
            else:
                m4_max_info["optimization_status"] = "not_available"
                
        except Exception as e:
            logger.warning(f"Error detecting M4 Max capabilities: {e}")
            m4_max_info["error"] = str(e)
        
        return m4_max_info
    
    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to all engine endpoints"""
        engines = {
            "analytics": "http://localhost:8100",
            "risk": "http://localhost:8200", 
            "factor": "http://localhost:8300",
            "ml": "http://localhost:8400",
            "features": "http://localhost:8500",
            "websocket": "http://localhost:8600",
            "strategy": "http://localhost:8700",
            "marketdata": "http://localhost:8800",
            "portfolio": "http://localhost:8900"
        }
        
        connectivity_status = {
            "reachable_engines": 0,
            "total_engines": len(engines),
            "engine_status": {}
        }
        
        for engine_name, url in engines.items():
            try:
                # Simple connectivity check using curl
                result = subprocess.run(
                    ["curl", "-s", "--connect-timeout", "2", f"{url}/health"],
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    connectivity_status["reachable_engines"] += 1
                    connectivity_status["engine_status"][engine_name] = "reachable"
                else:
                    connectivity_status["engine_status"][engine_name] = "unreachable"
                    
            except Exception as e:
                connectivity_status["engine_status"][engine_name] = f"error: {str(e)}"
        
        connectivity_status["connectivity_percentage"] = (
            connectivity_status["reachable_engines"] / connectivity_status["total_engines"]
        ) * 100
        
        return connectivity_status
    
    async def run_all_individual_engine_tests(self) -> Dict[str, Any]:
        """Run all individual engine performance tests"""
        logger.info("Running all individual engine performance tests...")
        
        individual_results = {
            "test_start_time": datetime.now().isoformat(),
            "tests_executed": {},
            "test_summary": {}
        }
        
        # Define test execution order (prioritize critical engines)
        test_order = [
            ("analytics_engine", "Analytics Engine"),
            ("risk_engine", "Risk Engine"),
            ("factor_engine", "Factor Engine"),  
            ("ml_engine", "ML Engine"),
            ("remaining_engines", "Remaining Engines (Features, WebSocket, Strategy, MarketData, Portfolio)")
        ]
        
        for test_key, test_description in test_order:
            logger.info(f"Executing {test_description} performance test...")
            test_start_time = time.time()
            
            try:
                test_instance = self.test_instances[test_key]
                test_result = await test_instance.run_comprehensive_test()
                
                execution_time = time.time() - test_start_time
                
                individual_results["tests_executed"][test_key] = {
                    "test_description": test_description,
                    "execution_time_seconds": execution_time,
                    "test_result": test_result,
                    "status": "completed"
                }
                
                logger.info(f"Completed {test_description} test in {execution_time:.2f} seconds ✅")
                
            except Exception as e:
                execution_time = time.time() - test_start_time
                
                individual_results["tests_executed"][test_key] = {
                    "test_description": test_description,
                    "execution_time_seconds": execution_time,
                    "error": str(e),
                    "status": "failed"
                }
                
                logger.error(f"Failed {test_description} test after {execution_time:.2f} seconds ❌ - {e}")
            
            # Brief pause between major test suites
            await asyncio.sleep(3)
        
        # Generate summary of individual tests
        completed_tests = [t for t in individual_results["tests_executed"].values() if t["status"] == "completed"]
        failed_tests = [t for t in individual_results["tests_executed"].values() if t["status"] == "failed"]
        
        individual_results["test_summary"] = {
            "total_tests": len(individual_results["tests_executed"]),
            "completed_tests": len(completed_tests),
            "failed_tests": len(failed_tests),
            "success_rate": (len(completed_tests) / len(individual_results["tests_executed"])) * 100,
            "total_execution_time": sum(t["execution_time_seconds"] for t in individual_results["tests_executed"].values()),
            "avg_execution_time": statistics.mean([t["execution_time_seconds"] for t in individual_results["tests_executed"].values()])
        }
        
        return individual_results
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        logger.info("Running comprehensive integration tests...")
        
        integration_start_time = time.time()
        
        try:
            integration_instance = self.test_instances["integration_tests"]
            integration_results = await integration_instance.run_comprehensive_test()
            
            execution_time = time.time() - integration_start_time
            
            return {
                "execution_time_seconds": execution_time,
                "test_results": integration_results,
                "status": "completed"
            }
            
        except Exception as e:
            execution_time = time.time() - integration_start_time
            
            logger.error(f"Integration tests failed after {execution_time:.2f} seconds: {e}")
            
            return {
                "execution_time_seconds": execution_time,
                "error": str(e),
                "status": "failed"
            }
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance from all test results"""
        logger.info("Analyzing overall system performance...")
        
        analysis = {
            "system_health_score": 0.0,
            "engine_performance_scores": {},
            "integration_score": 0.0,
            "m4_max_optimization_score": 0.0,
            "bottlenecks_identified": [],
            "performance_highlights": [],
            "critical_issues": []
        }
        
        # Analyze individual engine performance
        individual_results = self.test_results.get("individual_engine_tests", {})
        engine_scores = {}
        
        for test_key, test_data in individual_results.get("tests_executed", {}).items():
            if test_data.get("status") == "completed":
                test_result = test_data.get("test_result", {})
                performance_summary = test_result.get("performance_summary", {})
                
                # Extract performance metrics based on engine type
                if test_key == "analytics_engine":
                    score = self.calculate_analytics_score(performance_summary)
                elif test_key == "risk_engine":
                    score = self.calculate_risk_score(performance_summary)
                elif test_key == "factor_engine":
                    score = self.calculate_factor_score(performance_summary)
                elif test_key == "ml_engine":
                    score = self.calculate_ml_score(performance_summary)
                elif test_key == "remaining_engines":
                    score = self.calculate_remaining_engines_score(performance_summary)
                else:
                    score = 50.0  # Default neutral score
                
                engine_scores[test_key] = score
                analysis["engine_performance_scores"][test_key] = {
                    "score": score,
                    "health_status": performance_summary.get("overall_health", "unknown")
                }
        
        # Analyze integration performance
        integration_results = self.test_results.get("integration_tests", {})
        if integration_results.get("status") == "completed":
            integration_data = integration_results.get("test_results", {})
            integration_summary = integration_data.get("performance_summary", {})
            analysis["integration_score"] = self.calculate_integration_score(integration_summary)
        
        # Calculate overall system health score
        all_scores = list(engine_scores.values())
        if analysis["integration_score"] > 0:
            all_scores.append(analysis["integration_score"])
        
        if all_scores:
            analysis["system_health_score"] = statistics.mean(all_scores)
        
        # M4 Max optimization analysis
        analysis["m4_max_optimization_score"] = self.calculate_m4_max_score()
        
        # Identify bottlenecks and highlights
        analysis["bottlenecks_identified"] = self.identify_bottlenecks()
        analysis["performance_highlights"] = self.identify_highlights()
        analysis["critical_issues"] = self.identify_critical_issues()
        
        return analysis
    
    def calculate_analytics_score(self, summary: Dict[str, Any]) -> float:
        """Calculate performance score for Analytics Engine"""
        score = 50.0  # Base score
        
        # Health status
        health = summary.get("overall_health", "unknown")
        if health == "excellent":
            score += 30
        elif health == "good":
            score += 20
        elif health == "fair":
            score += 10
        
        # API performance
        api_perf = summary.get("api_response_performance", {})
        health_avg = api_perf.get("health_avg_ms", 1000)
        if health_avg < 50:
            score += 10
        elif health_avg < 100:
            score += 5
        
        # Scalability
        scalability = summary.get("scalability_metrics", {})
        success_rate = scalability.get("concurrent_success_rate", 0)
        if success_rate >= 95:
            score += 10
        elif success_rate >= 90:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def calculate_risk_score(self, summary: Dict[str, Any]) -> float:
        """Calculate performance score for Risk Engine"""
        score = 50.0  # Base score
        
        # ML integration status
        ml_status = summary.get("ml_integration_status", "unknown")
        if ml_status == "loaded":
            score += 20
        elif ml_status == "not_loaded":
            score -= 10
        
        # Risk management performance
        risk_perf = summary.get("risk_management_performance", {})
        throughput = risk_perf.get("throughput_checks_per_sec", 0)
        if throughput >= 10:
            score += 15
        elif throughput >= 5:
            score += 10
        elif throughput >= 1:
            score += 5
        
        # Analytics capabilities
        analytics_perf = summary.get("analytics_performance", {})
        if analytics_perf.get("pyfolio_response_time_ms", 0) > 0:
            score += 15  # PyFolio integration working
        
        return min(100.0, max(0.0, score))
    
    def calculate_factor_score(self, summary: Dict[str, Any]) -> float:
        """Calculate performance score for Factor Engine"""
        score = 50.0  # Base score
        
        # Factor definitions loaded
        system_metrics = summary.get("system_metrics", {})
        definitions_loaded = system_metrics.get("factor_definitions_loaded", 0)
        if definitions_loaded >= 400:  # Expected ~485
            score += 25
        elif definitions_loaded >= 300:
            score += 15
        elif definitions_loaded >= 100:
            score += 5
        
        # Processing performance
        processing_perf = summary.get("factor_processing_performance", {})
        calc_success_rate = processing_perf.get("calculation_success_rate", 0)
        if calc_success_rate >= 90:
            score += 15
        elif calc_success_rate >= 80:
            score += 10
        
        # Throughput
        throughput = system_metrics.get("factors_per_second", 0)
        if throughput >= 5:
            score += 10
        elif throughput >= 1:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def calculate_ml_score(self, summary: Dict[str, Any]) -> float:
        """Calculate performance score for ML Engine"""
        score = 50.0  # Base score
        
        # System performance
        system_perf = summary.get("system_performance", {})
        concurrent_success = system_perf.get("concurrent_success_rate", 0)
        if concurrent_success >= 90:
            score += 20
        elif concurrent_success >= 80:
            score += 15
        elif concurrent_success >= 70:
            score += 10
        
        # ML prediction performance
        ml_perf = summary.get("ml_prediction_performance", {})
        price_success = ml_perf.get("price_prediction_success_rate", 0)
        if price_success >= 90:
            score += 15
        elif price_success >= 80:
            score += 10
        
        # Model capabilities
        capabilities = summary.get("model_capabilities", {})
        regime_success = capabilities.get("regime_detection_success_rate", 0)
        vol_success = capabilities.get("volatility_forecasting_success_rate", 0)
        
        if regime_success >= 80 and vol_success >= 80:
            score += 15
        elif regime_success >= 70 or vol_success >= 70:
            score += 10
        
        return min(100.0, max(0.0, score))
    
    def calculate_remaining_engines_score(self, summary: Dict[str, Any]) -> float:
        """Calculate performance score for remaining engines"""
        score = 50.0  # Base score
        
        # Engine health summary
        engine_health = summary.get("engine_health_summary", {})
        health_percentage = engine_health.get("avg_success_rate", 0)
        
        if health_percentage >= 95:
            score += 25
        elif health_percentage >= 90:
            score += 20
        elif health_percentage >= 80:
            score += 15
        elif health_percentage >= 70:
            score += 10
        
        # Individual engine performance
        individual_perf = summary.get("individual_engine_performance", {})
        healthy_engines = 0
        total_engines = len(individual_perf)
        
        for engine_name, perf in individual_perf.items():
            if perf.get("success_rate", 0) >= 80:
                healthy_engines += 1
        
        if total_engines > 0:
            engine_health_ratio = healthy_engines / total_engines
            if engine_health_ratio >= 0.9:
                score += 15
            elif engine_health_ratio >= 0.8:
                score += 10
            elif engine_health_ratio >= 0.6:
                score += 5
        
        return min(100.0, max(0.0, score))
    
    def calculate_integration_score(self, summary: Dict[str, Any]) -> float:
        """Calculate integration performance score"""
        score = 50.0  # Base score
        
        # Overall integration health
        integration_health = summary.get("overall_integration_health", "unknown")
        if integration_health == "excellent":
            score += 30
        elif integration_health == "good":
            score += 20
        elif integration_health == "fair":
            score += 10
        
        # Engine health status
        engine_health = summary.get("engine_health_status", {})
        health_percentage = engine_health.get("health_percentage", 0)
        if health_percentage >= 90:
            score += 10
        elif health_percentage >= 80:
            score += 5
        
        # End-to-end metrics
        e2e_metrics = summary.get("end_to_end_metrics", {})
        e2e_success_rate = e2e_metrics.get("end_to_end_success_rate", 0)
        if e2e_success_rate >= 80:
            score += 10
        elif e2e_success_rate >= 70:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def calculate_m4_max_score(self) -> float:
        """Calculate M4 Max optimization score"""
        if not self.m4_max_metrics.get("detected", False):
            return 0.0  # Not running on M4 Max
        
        score = 20.0  # Base score for M4 Max detection
        
        # Metal GPU availability
        if self.m4_max_metrics.get("metal_gpu_available", False):
            score += 30
        
        # Unified memory detection
        if self.m4_max_metrics.get("unified_memory_detected", False):
            score += 20
        
        # CPU configuration
        cpu_config = self.m4_max_metrics.get("cpu_config", {})
        if cpu_config.get("performance_cores", 0) >= 10:
            score += 15
        
        if cpu_config.get("efficiency_cores", 0) >= 4:
            score += 10
        
        # Overall optimization status
        opt_status = self.m4_max_metrics.get("optimization_status", "not_available")
        if opt_status == "available":
            score += 5
        
        return min(100.0, score)
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify system bottlenecks from test results"""
        bottlenecks = []
        
        # Check individual engine performance for bottlenecks
        individual_results = self.test_results.get("individual_engine_tests", {})
        
        for test_key, test_data in individual_results.get("tests_executed", {}).items():
            if test_data.get("status") == "completed":
                test_result = test_data.get("test_result", {})
                performance_summary = test_result.get("performance_summary", {})
                
                # Check for performance issues
                if test_key == "factor_engine":
                    system_metrics = performance_summary.get("system_metrics", {})
                    if system_metrics.get("factor_definitions_loaded", 0) < 400:
                        bottlenecks.append({
                            "component": "Factor Engine",
                            "issue": "Insufficient factor definitions loaded",
                            "severity": "high",
                            "current_value": system_metrics.get("factor_definitions_loaded", 0),
                            "expected_value": "~485"
                        })
                
                elif test_key == "risk_engine":
                    ml_status = performance_summary.get("ml_integration_status", "unknown")
                    if ml_status != "loaded":
                        bottlenecks.append({
                            "component": "Risk Engine",
                            "issue": "ML model not loaded",
                            "severity": "medium",
                            "current_value": ml_status,
                            "expected_value": "loaded"
                        })
        
        # Check integration bottlenecks
        integration_results = self.test_results.get("integration_tests", {})
        if integration_results.get("status") == "completed":
            integration_data = integration_results.get("test_results", {})
            integration_summary = integration_data.get("performance_summary", {})
            
            e2e_metrics = integration_summary.get("end_to_end_metrics", {})
            e2e_success_rate = e2e_metrics.get("end_to_end_success_rate", 0)
            
            if e2e_success_rate < 70:
                bottlenecks.append({
                    "component": "System Integration",
                    "issue": "Low end-to-end success rate",
                    "severity": "high",
                    "current_value": f"{e2e_success_rate:.1f}%",
                    "expected_value": ">80%"
                })
        
        return bottlenecks
    
    def identify_highlights(self) -> List[Dict[str, Any]]:
        """Identify system performance highlights"""
        highlights = []
        
        # Check for high-performing components
        engine_scores = self.performance_summary.get("engine_performance_scores", {})
        
        for engine_name, score_data in engine_scores.items():
            score = score_data.get("score", 0)
            if score >= 85:
                highlights.append({
                    "component": engine_name.replace("_", " ").title(),
                    "achievement": f"Excellent performance score: {score:.1f}/100",
                    "category": "performance"
                })
        
        # M4 Max optimization highlights
        m4_max_score = self.performance_summary.get("m4_max_optimization_score", 0)
        if m4_max_score >= 80:
            highlights.append({
                "component": "M4 Max Hardware Optimization",
                "achievement": f"High optimization utilization: {m4_max_score:.1f}/100",
                "category": "hardware_acceleration"
            })
        
        # System health highlights
        system_health = self.performance_summary.get("system_health_score", 0)
        if system_health >= 80:
            highlights.append({
                "component": "Overall System Health",
                "achievement": f"Strong system performance: {system_health:.1f}/100",
                "category": "system_health"
            })
        
        return highlights
    
    def identify_critical_issues(self) -> List[Dict[str, Any]]:
        """Identify critical system issues requiring immediate attention"""
        critical_issues = []
        
        # Check for failed tests
        individual_results = self.test_results.get("individual_engine_tests", {})
        failed_tests = [
            test_data for test_data in individual_results.get("tests_executed", {}).values()
            if test_data.get("status") == "failed"
        ]
        
        if failed_tests:
            critical_issues.append({
                "issue": "Engine Test Failures",
                "description": f"{len(failed_tests)} engine test(s) failed to execute",
                "severity": "critical",
                "action_required": "Investigate and resolve failed engine tests",
                "failed_tests": [test.get("test_description", "unknown") for test in failed_tests]
            })
        
        # Check engine connectivity
        network_status = self.system_info.get("network_status", {})
        connectivity_percentage = network_status.get("connectivity_percentage", 0)
        
        if connectivity_percentage < 80:
            critical_issues.append({
                "issue": "Engine Connectivity Issues",
                "description": f"Only {connectivity_percentage:.1f}% of engines are reachable",
                "severity": "critical",
                "action_required": "Check Docker containers and network configuration"
            })
        
        # Check system health score
        system_health = self.performance_summary.get("system_health_score", 0)
        if system_health < 50:
            critical_issues.append({
                "issue": "Poor System Performance",
                "description": f"System health score is critically low: {system_health:.1f}/100",
                "severity": "critical",
                "action_required": "Review all engine performance issues and optimize system configuration"
            })
        
        return critical_issues
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate comprehensive system recommendations"""
        recommendations = []
        
        # M4 Max optimization recommendations
        if self.m4_max_metrics.get("detected", False):
            m4_max_score = self.performance_summary.get("m4_max_optimization_score", 0)
            
            if m4_max_score < 80:
                recommendations.append({
                    "category": "hardware_optimization",
                    "priority": "high",
                    "title": "Enable M4 Max Hardware Acceleration",
                    "description": "Full M4 Max optimization not detected",
                    "action": "Enable Metal GPU, Neural Engine, and unified memory optimizations for 50x+ performance improvements",
                    "expected_improvement": "50-71x performance boost in trading operations"
                })
        
        # Engine-specific recommendations
        bottlenecks = self.performance_summary.get("bottlenecks_identified", [])
        for bottleneck in bottlenecks:
            if bottleneck.get("severity") == "high":
                recommendations.append({
                    "category": "engine_optimization",
                    "priority": "high",
                    "title": f"Fix {bottleneck.get('component')} Performance",
                    "description": bottleneck.get("issue"),
                    "action": f"Address {bottleneck.get('issue')} - current: {bottleneck.get('current_value')}, expected: {bottleneck.get('expected_value')}"
                })
        
        # Integration improvements
        integration_score = self.performance_summary.get("integration_score", 0)
        if integration_score < 70:
            recommendations.append({
                "category": "integration",
                "priority": "medium",
                "title": "Improve Inter-Engine Communication",
                "description": f"Integration score is {integration_score:.1f}/100",
                "action": "Optimize MessageBus configuration, review timeout settings, and enhance error handling"
            })
        
        # System scaling recommendations
        system_health = self.performance_summary.get("system_health_score", 0)
        if system_health >= 80:
            recommendations.append({
                "category": "scaling",
                "priority": "low",
                "title": "Consider Horizontal Scaling",
                "description": "System is performing well and ready for scaling",
                "action": "Implement engine scaling strategies to handle increased load"
            })
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system performance report"""
        logger.info("Generating comprehensive system performance report...")
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "test_duration_seconds": (datetime.now() - self.test_start_time).total_seconds(),
                "report_version": "1.0.0",
                "system_under_test": "Nautilus Containerized Trading Platform"
            },
            "executive_summary": self.generate_executive_summary(),
            "system_information": self.system_info,
            "m4_max_capabilities": self.m4_max_metrics,
            "test_results": self.test_results,
            "performance_analysis": self.performance_summary,
            "recommendations": self.recommendations,
            "detailed_findings": self.generate_detailed_findings(),
            "appendix": self.generate_appendix()
        }
        
        return report
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of system performance"""
        system_health = self.performance_summary.get("system_health_score", 0)
        m4_max_score = self.performance_summary.get("m4_max_optimization_score", 0)
        
        # Determine overall system grade
        if system_health >= 90:
            grade = "A (Excellent)"
        elif system_health >= 80:
            grade = "B (Good)"
        elif system_health >= 70:
            grade = "C (Fair)"
        elif system_health >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Critical)"
        
        summary = {
            "overall_grade": grade,
            "system_health_score": system_health,
            "m4_max_optimization_score": m4_max_score,
            "key_findings": [
                f"System achieved {system_health:.1f}/100 overall health score",
                f"M4 Max optimization score: {m4_max_score:.1f}/100",
                f"{len(self.performance_summary.get('bottlenecks_identified', []))} bottlenecks identified",
                f"{len(self.performance_summary.get('critical_issues', []))} critical issues require attention"
            ],
            "engine_status_summary": {},
            "integration_status": "unknown"
        }
        
        # Engine status summary
        engine_scores = self.performance_summary.get("engine_performance_scores", {})
        healthy_engines = len([s for s in engine_scores.values() if s.get("score", 0) >= 70])
        total_engines = len(engine_scores)
        
        summary["engine_status_summary"] = {
            "healthy_engines": healthy_engines,
            "total_engines": total_engines,
            "health_percentage": (healthy_engines / total_engines * 100) if total_engines > 0 else 0
        }
        
        # Integration status
        integration_score = self.performance_summary.get("integration_score", 0)
        if integration_score >= 80:
            summary["integration_status"] = "excellent"
        elif integration_score >= 70:
            summary["integration_status"] = "good"
        elif integration_score >= 60:
            summary["integration_status"] = "fair"
        else:
            summary["integration_status"] = "poor"
        
        return summary
    
    def generate_detailed_findings(self) -> Dict[str, Any]:
        """Generate detailed findings for each component"""
        findings = {
            "individual_engine_analysis": {},
            "integration_analysis": {},
            "performance_trends": {},
            "optimization_opportunities": []
        }
        
        # Individual engine analysis
        individual_results = self.test_results.get("individual_engine_tests", {})
        for test_key, test_data in individual_results.get("tests_executed", {}).items():
            if test_data.get("status") == "completed":
                findings["individual_engine_analysis"][test_key] = {
                    "execution_time": test_data.get("execution_time_seconds", 0),
                    "performance_summary": test_data.get("test_result", {}).get("performance_summary", {}),
                    "key_metrics": self.extract_key_metrics(test_key, test_data.get("test_result", {}))
                }
        
        # Integration analysis
        integration_results = self.test_results.get("integration_tests", {})
        if integration_results.get("status") == "completed":
            findings["integration_analysis"] = {
                "execution_time": integration_results.get("execution_time_seconds", 0),
                "performance_summary": integration_results.get("test_results", {}).get("performance_summary", {}),
                "flow_analysis": self.analyze_integration_flows(integration_results.get("test_results", {}))
            }
        
        return findings
    
    def extract_key_metrics(self, engine_name: str, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for each engine type"""
        performance_summary = test_result.get("performance_summary", {})
        
        if engine_name == "analytics_engine":
            return {
                "health_response_time_ms": performance_summary.get("api_response_performance", {}).get("health_avg_ms", 0),
                "concurrent_success_rate": performance_summary.get("scalability_metrics", {}).get("concurrent_success_rate", 0),
                "throughput_ops_per_sec": performance_summary.get("api_response_performance", {}).get("performance_calc_throughput", 0)
            }
        elif engine_name == "factor_engine":
            return {
                "factor_definitions_loaded": performance_summary.get("system_metrics", {}).get("factor_definitions_loaded", 0),
                "factors_per_second": performance_summary.get("system_metrics", {}).get("factors_per_second", 0),
                "cache_hit_rate": performance_summary.get("system_metrics", {}).get("cache_hit_rate", 0)
            }
        # Add more engine-specific metrics as needed
        else:
            return {"overall_health": performance_summary.get("overall_health", "unknown")}
    
    def analyze_integration_flows(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze integration flow performance"""
        flow_analysis = {}
        
        performance_summary = integration_result.get("performance_summary", {})
        flow_perf = performance_summary.get("integration_flow_performance", {})
        
        for flow_name, metrics in flow_perf.items():
            flow_analysis[flow_name] = {
                "success_rate": metrics.get("success_rate", 0),
                "steps_completed": metrics.get("steps_completed", 0),
                "total_steps": metrics.get("total_steps", 0),
                "performance_rating": "excellent" if metrics.get("success_rate", 0) >= 90 else 
                                   "good" if metrics.get("success_rate", 0) >= 80 else
                                   "fair" if metrics.get("success_rate", 0) >= 70 else "poor"
            }
        
        return flow_analysis
    
    def generate_appendix(self) -> Dict[str, Any]:
        """Generate appendix with detailed technical information"""
        return {
            "test_configuration": {
                "engines_tested": list(self.test_instances.keys()),
                "test_data_sources": ["YFinance", "Synthetic Market Data", "Mock Economic Data"],
                "test_methodology": "Comprehensive performance testing with real-world scenarios"
            },
            "hardware_specifications": self.m4_max_metrics,
            "docker_environment": self.system_info.get("docker_status", {}),
            "network_configuration": self.system_info.get("network_status", {}),
            "performance_benchmarks": {
                "target_response_times": {
                    "health_endpoints": "< 100ms",
                    "analytics_calculations": "< 500ms",
                    "risk_checks": "< 200ms",
                    "factor_calculations": "< 1000ms"
                },
                "target_throughput": {
                    "concurrent_requests": "> 10 requests/second",
                    "factor_calculations": "> 5 factors/second",
                    "risk_checks": "> 10 checks/second"
                }
            }
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info("="*100)
        logger.info("STARTING COMPREHENSIVE SYSTEM PERFORMANCE TEST SUITE")
        logger.info("="*100)
        
        overall_start_time = time.time()
        
        try:
            # Phase 1: Run individual engine tests
            logger.info("PHASE 1: Running individual engine performance tests...")
            self.test_results["individual_engine_tests"] = await self.run_all_individual_engine_tests()
            
            # Brief pause between phases
            await asyncio.sleep(5)
            
            # Phase 2: Run integration tests
            logger.info("PHASE 2: Running integration tests...")
            self.test_results["integration_tests"] = await self.run_integration_tests()
            
            # Phase 3: Analyze system performance
            logger.info("PHASE 3: Analyzing system performance...")
            self.performance_summary = self.analyze_system_performance()
            
            # Phase 4: Generate recommendations
            logger.info("PHASE 4: Generating recommendations...")
            self.recommendations = self.generate_recommendations()
            
            # Phase 5: Generate comprehensive report
            logger.info("PHASE 5: Generating comprehensive report...")
            comprehensive_report = self.generate_comprehensive_report()
            
            total_execution_time = time.time() - overall_start_time
            comprehensive_report["report_metadata"]["total_execution_time_seconds"] = total_execution_time
            
            logger.info("="*100)
            logger.info("COMPREHENSIVE SYSTEM PERFORMANCE TEST SUITE COMPLETE")
            logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
            logger.info(f"System Health Score: {self.performance_summary.get('system_health_score', 0):.1f}/100")
            logger.info(f"M4 Max Optimization Score: {self.performance_summary.get('m4_max_optimization_score', 0):.1f}/100")
            logger.info("="*100)
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Comprehensive test suite failed: {e}")
            
            # Return partial results if available
            partial_report = {
                "error": str(e),
                "partial_results": {
                    "system_info": self.system_info,
                    "m4_max_metrics": self.m4_max_metrics,
                    "test_results": self.test_results,
                    "performance_summary": self.performance_summary
                },
                "execution_time_seconds": time.time() - overall_start_time
            }
            
            return partial_report

    def save_report_to_file(self, report: Dict[str, Any]):
        """Save comprehensive report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/comprehensive_system_performance_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Comprehensive report saved to: {filename}")
            
            # Also save a summary report
            summary_filename = f"/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/tests/performance_summary_{timestamp}.json"
            summary_report = {
                "executive_summary": report.get("executive_summary", {}),
                "performance_analysis": report.get("performance_analysis", {}),
                "recommendations": report.get("recommendations", []),
                "system_health_score": self.performance_summary.get("system_health_score", 0),
                "m4_max_optimization_score": self.performance_summary.get("m4_max_optimization_score", 0)
            }
            
            with open(summary_filename, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            logger.info(f"Performance summary saved to: {summary_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save report to file: {e}")

def print_executive_summary(report: Dict[str, Any]):
    """Print executive summary to console"""
    exec_summary = report.get("executive_summary", {})
    performance_analysis = report.get("performance_analysis", {})
    
    print("\n" + "="*100)
    print("NAUTILUS CONTAINERIZED TRADING PLATFORM - PERFORMANCE TEST RESULTS")
    print("="*100)
    
    print(f"\n📊 OVERALL SYSTEM GRADE: {exec_summary.get('overall_grade', 'N/A')}")
    print(f"🏥 System Health Score: {exec_summary.get('system_health_score', 0):.1f}/100")
    print(f"🚀 M4 Max Optimization Score: {exec_summary.get('m4_max_optimization_score', 0):.1f}/100")
    
    # Engine status
    engine_summary = exec_summary.get("engine_status_summary", {})
    print(f"\n🔧 ENGINE STATUS:")
    print(f"   Healthy Engines: {engine_summary.get('healthy_engines', 0)}/{engine_summary.get('total_engines', 0)}")
    print(f"   Engine Health: {engine_summary.get('health_percentage', 0):.1f}%")
    
    # Integration status
    print(f"🔗 Integration Status: {exec_summary.get('integration_status', 'unknown').upper()}")
    
    # Key findings
    key_findings = exec_summary.get("key_findings", [])
    if key_findings:
        print(f"\n📋 KEY FINDINGS:")
        for i, finding in enumerate(key_findings, 1):
            print(f"   {i}. {finding}")
    
    # Critical issues
    critical_issues = performance_analysis.get("critical_issues", [])
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
        for i, issue in enumerate(critical_issues, 1):
            print(f"   {i}. {issue.get('issue', 'Unknown')}: {issue.get('description', 'No description')}")
    
    # Performance highlights
    highlights = performance_analysis.get("performance_highlights", [])
    if highlights:
        print(f"\n✨ PERFORMANCE HIGHLIGHTS ({len(highlights)}):")
        for i, highlight in enumerate(highlights, 1):
            print(f"   {i}. {highlight.get('component', 'Unknown')}: {highlight.get('achievement', 'No details')}")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]
    if high_priority_recs:
        print(f"\n💡 HIGH PRIORITY RECOMMENDATIONS ({len(high_priority_recs)}):")
        for i, rec in enumerate(high_priority_recs, 1):
            print(f"   {i}. {rec.get('title', 'Unknown')}")
            print(f"      Action: {rec.get('action', 'No action specified')}")
    
    print("="*100)

async def main():
    """Main execution function"""
    # Create comprehensive test suite
    test_suite = ComprehensiveSystemPerformanceTest()
    
    # Run complete test suite
    report = await test_suite.run_comprehensive_test_suite()
    
    # Save comprehensive report
    test_suite.save_report_to_file(report)
    
    # Print executive summary
    print_executive_summary(report)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
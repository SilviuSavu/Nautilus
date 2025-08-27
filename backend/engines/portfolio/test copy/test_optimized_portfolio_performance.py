#!/usr/bin/env python3
"""
Optimized Portfolio Engine Performance Tests & Validation
Comprehensive testing suite for the optimized portfolio engine with SME acceleration
and dual messagebus integration.
"""

import asyncio
import pytest
import time
import json
import statistics
import sys
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Add backend to path
sys.path.append('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend')

# Import the optimized portfolio engine
from optimized_portfolio_engine import OptimizedPortfolioEngine, OptimizedPortfolio
from portfolio_monitoring import comprehensive_monitor, MetricType

# Test configuration
TEST_CONFIG = {
    "performance_targets": {
        "portfolio_creation_ms": 10.0,
        "optimization_ms": 5.0,
        "risk_analysis_ms": 3.0,
        "messagebus_latency_ms": 2.0,
        "sme_speedup_factor": 5.0
    },
    "load_test": {
        "concurrent_requests": 50,
        "total_operations": 1000,
        "duration_minutes": 5
    },
    "stress_test": {
        "max_portfolios": 500,
        "batch_size": 10,
        "operation_types": ["create", "optimize", "risk_analysis", "rebalance"]
    }
}

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.engine = None
        self.test_results: Dict[str, List[float]] = {
            "portfolio_creation": [],
            "optimization": [],
            "risk_analysis": [],
            "messagebus_operations": [],
            "sme_operations": []
        }
        self.test_portfolios: List[str] = []
    
    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up performance test environment...")
        
        # Initialize optimized portfolio engine
        self.engine = OptimizedPortfolioEngine()
        await self.engine.initialize()
        
        # Initialize monitoring
        await comprehensive_monitor.initialize()
        
        print("✅ Test environment ready")
    
    async def teardown(self):
        """Teardown test environment"""
        print("🔄 Tearing down test environment...")
        
        # Clean up test portfolios
        for portfolio_id in self.test_portfolios:
            if portfolio_id in self.engine.portfolios:
                del self.engine.portfolios[portfolio_id]
        
        # Stop engine
        await self.engine.stop()
        
        # Shutdown monitoring
        await comprehensive_monitor.shutdown()
        
        print("✅ Test environment cleaned up")
    
    async def test_portfolio_creation_performance(self, num_portfolios: int = 100) -> Dict[str, Any]:
        """Test portfolio creation performance"""
        print(f"📊 Testing portfolio creation performance ({num_portfolios} portfolios)...")
        
        creation_times = []
        successful_creations = 0
        
        for i in range(num_portfolios):
            config = {
                "name": f"Performance Test Portfolio {i}",
                "tier": "institutional",
                "initial_capital": 1000000 + (i * 100000),
                "client_id": f"test_client_{i}",
                "initial_positions": {
                    "SPY": {"quantity": 100, "avg_cost": 400},
                    "QQQ": {"quantity": 50, "avg_cost": 300},
                    "IWM": {"quantity": 25, "avg_cost": 200}
                }
            }
            
            start_time = time.perf_counter()
            
            try:
                result = await self.engine.create_institutional_portfolio(config)
                portfolio_id = result["portfolio_id"]
                self.test_portfolios.append(portfolio_id)
                
                creation_time_ms = (time.perf_counter() - start_time) * 1000
                creation_times.append(creation_time_ms)
                successful_creations += 1
                
                # Record performance metric
                comprehensive_monitor.record_request_performance(
                    "portfolio_creation", creation_time_ms, True
                )
                
            except Exception as e:
                creation_time_ms = (time.perf_counter() - start_time) * 1000
                creation_times.append(creation_time_ms)
                print(f"❌ Portfolio creation failed: {e}")
                
                comprehensive_monitor.record_request_performance(
                    "portfolio_creation", creation_time_ms, False
                )
        
        # Calculate statistics
        avg_time = statistics.mean(creation_times)
        median_time = statistics.median(creation_times)
        p95_time = statistics.quantiles(creation_times, n=20)[18] if len(creation_times) >= 20 else max(creation_times)
        min_time = min(creation_times)
        max_time = max(creation_times)
        
        target_met = avg_time <= TEST_CONFIG["performance_targets"]["portfolio_creation_ms"]
        
        result = {
            "test_name": "Portfolio Creation Performance",
            "portfolios_tested": num_portfolios,
            "successful_creations": successful_creations,
            "success_rate": successful_creations / num_portfolios,
            "performance_stats": {
                "average_ms": avg_time,
                "median_ms": median_time,
                "p95_ms": p95_time,
                "min_ms": min_time,
                "max_ms": max_time
            },
            "target_ms": TEST_CONFIG["performance_targets"]["portfolio_creation_ms"],
            "target_met": target_met,
            "grade": "A+" if target_met and avg_time < 5.0 else "A" if target_met else "B",
            "sme_acceleration_active": self.engine.sme_accelerator.initialized
        }
        
        self.test_results["portfolio_creation"] = creation_times
        return result
    
    async def test_portfolio_optimization_performance(self, num_optimizations: int = 50) -> Dict[str, Any]:
        """Test portfolio optimization performance"""
        print(f"🚀 Testing portfolio optimization performance ({num_optimizations} optimizations)...")
        
        # Ensure we have test portfolios
        if len(self.test_portfolios) < num_optimizations:
            await self.test_portfolio_creation_performance(num_optimizations)
        
        optimization_times = []
        successful_optimizations = 0
        sme_accelerated_count = 0
        
        for i in range(min(num_optimizations, len(self.test_portfolios))):
            portfolio_id = self.test_portfolios[i]
            
            optimization_config = {
                "method": "optimization",
                "risk_tolerance": "moderate",
                "optimization_objective": "max_sharpe",
                "constraints": {
                    "max_weight": 0.4,
                    "min_weight": 0.05
                }
            }
            
            start_time = time.perf_counter()
            
            try:
                result = await self.engine.optimize_portfolio(portfolio_id, optimization_config)
                
                optimization_time_ms = (time.perf_counter() - start_time) * 1000
                optimization_times.append(optimization_time_ms)
                successful_optimizations += 1
                
                if result.sme_acceleration_used:
                    sme_accelerated_count += 1
                
                # Record performance metrics
                comprehensive_monitor.record_request_performance(
                    "portfolio_optimization", optimization_time_ms, True
                )
                
                comprehensive_monitor.record_sme_performance(
                    "portfolio_optimization",
                    result.calculation_time_nanoseconds,
                    result.sme_acceleration_used
                )
                
            except Exception as e:
                optimization_time_ms = (time.perf_counter() - start_time) * 1000
                optimization_times.append(optimization_time_ms)
                print(f"❌ Portfolio optimization failed: {e}")
                
                comprehensive_monitor.record_request_performance(
                    "portfolio_optimization", optimization_time_ms, False
                )
        
        # Calculate statistics
        avg_time = statistics.mean(optimization_times) if optimization_times else 0
        median_time = statistics.median(optimization_times) if optimization_times else 0
        p95_time = statistics.quantiles(optimization_times, n=20)[18] if len(optimization_times) >= 20 else (max(optimization_times) if optimization_times else 0)
        
        target_met = avg_time <= TEST_CONFIG["performance_targets"]["optimization_ms"]
        sme_utilization = sme_accelerated_count / successful_optimizations if successful_optimizations > 0 else 0
        
        result = {
            "test_name": "Portfolio Optimization Performance",
            "optimizations_tested": num_optimizations,
            "successful_optimizations": successful_optimizations,
            "success_rate": successful_optimizations / num_optimizations if num_optimizations > 0 else 0,
            "performance_stats": {
                "average_ms": avg_time,
                "median_ms": median_time,
                "p95_ms": p95_time,
                "min_ms": min(optimization_times) if optimization_times else 0,
                "max_ms": max(optimization_times) if optimization_times else 0
            },
            "target_ms": TEST_CONFIG["performance_targets"]["optimization_ms"],
            "target_met": target_met,
            "sme_acceleration": {
                "utilization_rate": sme_utilization,
                "accelerated_operations": sme_accelerated_count,
                "speedup_achieved": sme_utilization > 0.8
            },
            "grade": "A+" if target_met and sme_utilization > 0.8 else "A" if target_met else "B"
        }
        
        self.test_results["optimization"] = optimization_times
        return result
    
    async def test_risk_analysis_performance(self, num_analyses: int = 50) -> Dict[str, Any]:
        """Test risk analysis performance"""
        print(f"📈 Testing risk analysis performance ({num_analyses} analyses)...")
        
        # Ensure we have test portfolios
        if len(self.test_portfolios) < num_analyses:
            await self.test_portfolio_creation_performance(num_analyses)
        
        analysis_times = []
        successful_analyses = 0
        
        for i in range(min(num_analyses, len(self.test_portfolios))):
            portfolio_id = self.test_portfolios[i]
            
            start_time = time.perf_counter()
            
            try:
                result = await self.engine.run_portfolio_risk_analysis(portfolio_id)
                
                analysis_time_ms = (time.perf_counter() - start_time) * 1000
                analysis_times.append(analysis_time_ms)
                successful_analyses += 1
                
                comprehensive_monitor.record_request_performance(
                    "risk_analysis", analysis_time_ms, True
                )
                
            except Exception as e:
                analysis_time_ms = (time.perf_counter() - start_time) * 1000
                analysis_times.append(analysis_time_ms)
                print(f"❌ Risk analysis failed: {e}")
                
                comprehensive_monitor.record_request_performance(
                    "risk_analysis", analysis_time_ms, False
                )
        
        # Calculate statistics
        avg_time = statistics.mean(analysis_times) if analysis_times else 0
        target_met = avg_time <= TEST_CONFIG["performance_targets"]["risk_analysis_ms"]
        
        result = {
            "test_name": "Risk Analysis Performance",
            "analyses_tested": num_analyses,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / num_analyses if num_analyses > 0 else 0,
            "performance_stats": {
                "average_ms": avg_time,
                "median_ms": statistics.median(analysis_times) if analysis_times else 0,
                "p95_ms": statistics.quantiles(analysis_times, n=20)[18] if len(analysis_times) >= 20 else (max(analysis_times) if analysis_times else 0)
            },
            "target_ms": TEST_CONFIG["performance_targets"]["risk_analysis_ms"],
            "target_met": target_met,
            "grade": "A+" if target_met and avg_time < 2.0 else "A" if target_met else "B"
        }
        
        self.test_results["risk_analysis"] = analysis_times
        return result
    
    async def test_messagebus_performance(self, num_operations: int = 100) -> Dict[str, Any]:
        """Test messagebus performance"""
        print(f"💬 Testing messagebus performance ({num_operations} operations)...")
        
        if not self.engine.messagebus_connected:
            return {
                "test_name": "MessageBus Performance",
                "status": "SKIPPED",
                "reason": "MessageBus not available",
                "grade": "N/A"
            }
        
        latencies = []
        successful_operations = 0
        
        for i in range(num_operations):
            start_time = time.perf_counter()
            
            try:
                # Test messagebus notification
                await self.engine._notify_portfolio_creation(
                    f"test_portfolio_{i}",
                    {"test": True, "iteration": i}
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
                successful_operations += 1
                
                comprehensive_monitor.record_messagebus_latency(
                    "notification", latency_ms
                )
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
                print(f"❌ MessageBus operation failed: {e}")
        
        avg_latency = statistics.mean(latencies) if latencies else 0
        target_met = avg_latency <= TEST_CONFIG["performance_targets"]["messagebus_latency_ms"]
        
        result = {
            "test_name": "MessageBus Performance",
            "operations_tested": num_operations,
            "successful_operations": successful_operations,
            "success_rate": successful_operations / num_operations if num_operations > 0 else 0,
            "performance_stats": {
                "average_latency_ms": avg_latency,
                "median_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0)
            },
            "target_ms": TEST_CONFIG["performance_targets"]["messagebus_latency_ms"],
            "target_met": target_met,
            "dual_bus_active": self.engine.messagebus_connected,
            "grade": "A+" if target_met and avg_latency < 1.0 else "A" if target_met else "B"
        }
        
        self.test_results["messagebus_operations"] = latencies
        return result
    
    async def test_concurrent_load(self, concurrent_requests: int = 50, 
                                  total_operations: int = 1000) -> Dict[str, Any]:
        """Test concurrent load handling"""
        print(f"⚡ Testing concurrent load ({concurrent_requests} concurrent, {total_operations} total)...")
        
        async def create_portfolio_task(task_id: int) -> Tuple[bool, float]:
            """Single portfolio creation task"""
            config = {
                "name": f"Load Test Portfolio {task_id}",
                "tier": "institutional",
                "initial_capital": 1000000,
                "client_id": f"load_test_client_{task_id}"
            }
            
            start_time = time.perf_counter()
            
            try:
                result = await self.engine.create_institutional_portfolio(config)
                portfolio_id = result["portfolio_id"]
                self.test_portfolios.append(portfolio_id)
                
                execution_time = (time.perf_counter() - start_time) * 1000
                return True, execution_time
                
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                print(f"❌ Load test task {task_id} failed: {e}")
                return False, execution_time
        
        # Execute concurrent operations
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_task(task_id: int) -> Tuple[bool, float]:
            async with semaphore:
                return await create_portfolio_task(task_id)
        
        start_time = time.perf_counter()
        
        # Run all tasks concurrently
        tasks = [limited_task(i) for i in range(total_operations)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_operations = sum(1 for success, _ in results if success)
        execution_times = [time_ms for _, time_ms in results]
        
        throughput_ops_per_sec = total_operations / total_time
        avg_response_time = statistics.mean(execution_times)
        success_rate = successful_operations / total_operations
        
        result = {
            "test_name": "Concurrent Load Test",
            "concurrent_requests": concurrent_requests,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": success_rate,
            "performance_stats": {
                "total_time_seconds": total_time,
                "throughput_ops_per_sec": throughput_ops_per_sec,
                "average_response_time_ms": avg_response_time,
                "p95_response_time_ms": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times)
            },
            "grade": "A+" if success_rate > 0.98 and throughput_ops_per_sec > 100 else "A" if success_rate > 0.95 else "B"
        }
        
        return result
    
    async def test_memory_efficiency(self, max_portfolios: int = 500) -> Dict[str, Any]:
        """Test memory efficiency with large number of portfolios"""
        print(f"💾 Testing memory efficiency ({max_portfolios} portfolios)...")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        created_portfolios = 0
        memory_measurements = []
        
        for batch_start in range(0, max_portfolios, 50):  # Create in batches of 50
            batch_end = min(batch_start + 50, max_portfolios)
            
            # Create batch of portfolios
            for i in range(batch_start, batch_end):
                config = {
                    "name": f"Memory Test Portfolio {i}",
                    "tier": "institutional",
                    "initial_capital": 1000000,
                    "client_id": f"memory_test_client_{i}",
                    "initial_positions": {
                        f"STOCK_{i%10}": {"quantity": 100, "avg_cost": 50 + i}
                    }
                }
                
                try:
                    result = await self.engine.create_institutional_portfolio(config)
                    portfolio_id = result["portfolio_id"]
                    self.test_portfolios.append(portfolio_id)
                    created_portfolios += 1
                except Exception as e:
                    print(f"❌ Memory test portfolio creation failed: {e}")
            
            # Measure memory after each batch
            gc.collect()  # Force garbage collection
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_measurements.append({
                "portfolios": created_portfolios,
                "memory_mb": current_memory_mb,
                "memory_increase_mb": current_memory_mb - initial_memory_mb
            })
        
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_per_portfolio_kb = ((final_memory_mb - initial_memory_mb) * 1024) / created_portfolios if created_portfolios > 0 else 0
        
        result = {
            "test_name": "Memory Efficiency Test",
            "portfolios_created": created_portfolios,
            "memory_stats": {
                "initial_memory_mb": initial_memory_mb,
                "final_memory_mb": final_memory_mb,
                "total_increase_mb": final_memory_mb - initial_memory_mb,
                "memory_per_portfolio_kb": memory_per_portfolio_kb
            },
            "memory_measurements": memory_measurements,
            "efficient": memory_per_portfolio_kb < 100,  # Less than 100KB per portfolio
            "grade": "A+" if memory_per_portfolio_kb < 50 else "A" if memory_per_portfolio_kb < 100 else "B"
        }
        
        return result
    
    async def run_comprehensive_performance_suite(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        print("🚀 STARTING COMPREHENSIVE PORTFOLIO ENGINE PERFORMANCE SUITE")
        print("=" * 80)
        
        suite_start_time = time.perf_counter()
        test_results = {}
        
        try:
            # Run individual performance tests
            test_results["portfolio_creation"] = await self.test_portfolio_creation_performance(100)
            test_results["optimization"] = await self.test_portfolio_optimization_performance(50)
            test_results["risk_analysis"] = await self.test_risk_analysis_performance(50)
            test_results["messagebus"] = await self.test_messagebus_performance(100)
            test_results["concurrent_load"] = await self.test_concurrent_load(25, 250)
            test_results["memory_efficiency"] = await self.test_memory_efficiency(200)
            
            # Calculate overall grade
            grades = [test["grade"] for test in test_results.values() if "grade" in test and test["grade"] != "N/A"]
            grade_scores = {"A+": 4, "A": 3, "B": 2, "C": 1}
            avg_score = statistics.mean([grade_scores.get(grade, 1) for grade in grades]) if grades else 1
            
            if avg_score >= 3.5:
                overall_grade = "A+"
            elif avg_score >= 2.5:
                overall_grade = "A"
            else:
                overall_grade = "B"
            
            suite_duration = time.perf_counter() - suite_start_time
            
            # Compile comprehensive results
            comprehensive_results = {
                "test_suite": "Optimized Portfolio Engine Performance Suite",
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": suite_duration,
                "engine_info": {
                    "sme_acceleration": self.engine.sme_accelerator.initialized,
                    "messagebus_connected": self.engine.messagebus_connected,
                    "portfolios_tested": len(self.test_portfolios)
                },
                "individual_tests": test_results,
                "overall_performance": {
                    "grade": overall_grade,
                    "tests_passed": len([t for t in test_results.values() if t.get("target_met", False)]),
                    "total_tests": len(test_results),
                    "success_rate": len([t for t in test_results.values() if t.get("target_met", False)]) / len(test_results)
                },
                "monitoring_summary": comprehensive_monitor.get_comprehensive_status()
            }
            
            return comprehensive_results
            
        except Exception as e:
            print(f"❌ Performance suite failed: {e}")
            return {
                "test_suite": "Optimized Portfolio Engine Performance Suite",
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted results summary"""
        print("\n" + "=" * 80)
        print("🏆 PERFORMANCE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        if "overall_performance" in results:
            overall = results["overall_performance"]
            print(f"📊 Overall Grade: {overall['grade']}")
            print(f"✅ Tests Passed: {overall['tests_passed']}/{overall['total_tests']}")
            print(f"📈 Success Rate: {overall['success_rate']:.1%}")
        
        if "individual_tests" in results:
            print("\n📋 Individual Test Results:")
            for test_name, test_data in results["individual_tests"].items():
                if isinstance(test_data, dict) and "grade" in test_data:
                    status = "✅" if test_data.get("target_met", False) else "⚠️"
                    print(f"{status} {test_data['test_name']}: Grade {test_data['grade']}")
        
        if "engine_info" in results:
            engine = results["engine_info"]
            print(f"\n🔧 Engine Configuration:")
            print(f"   SME Acceleration: {'✅ Active' if engine['sme_acceleration'] else '❌ Inactive'}")
            print(f"   MessageBus: {'✅ Connected' if engine['messagebus_connected'] else '❌ Standalone'}")
            print(f"   Portfolios Tested: {engine['portfolios_tested']}")
        
        print("=" * 80)

async def main():
    """Main test execution function"""
    # Create and run performance test suite
    test_suite = PerformanceTestSuite()
    
    try:
        # Setup test environment
        await test_suite.setup()
        
        # Run comprehensive performance tests
        results = await test_suite.run_comprehensive_performance_suite()
        
        # Print results
        test_suite.print_results_summary(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optimized_portfolio_performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📝 Detailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return None
    
    finally:
        # Cleanup
        await test_suite.teardown()

if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run the performance test suite
    results = asyncio.run(main())
    
    # Exit with appropriate code
    if results and results.get("overall_performance", {}).get("success_rate", 0) > 0.8:
        print("🎉 Performance tests PASSED!")
        sys.exit(0)
    else:
        print("❌ Performance tests FAILED!")
        sys.exit(1)
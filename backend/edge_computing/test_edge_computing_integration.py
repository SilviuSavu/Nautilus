"""
Integration Test Suite for Edge Computing Systems

This module provides comprehensive integration tests for all edge computing
components to validate functionality and performance.
"""

import asyncio
import logging
import pytest
import time
from typing import Dict, List, Any

from .edge_node_manager import (
    EdgeNodeManager, EdgeNodeSpec, EdgeDeploymentConfig,
    TradingRegion, EdgeNodeType, NodeStatus
)
from .edge_placement_optimizer import (
    EdgePlacementOptimizer, PlacementStrategy, TradingActivityPattern,
    TradingActivityType
)
from .edge_cache_manager import (
    EdgeCacheManager, CacheConfiguration, CacheStrategy,
    ConsistencyLevel, DataCategory
)
from .regional_performance_optimizer import (
    RegionalPerformanceOptimizer, RegionalPerformanceProfile,
    OptimizationObjective, PerformanceTier
)
from .edge_failover_manager import (
    EdgeFailoverManager, FailoverConfiguration, FailoverStrategy,
    ConsistencyModel
)
from .edge_monitoring_system import (
    EdgeMonitoringSystem, AlertRule, AlertSeverity, MetricType
)


class EdgeComputingIntegrationTest:
    """Comprehensive integration test suite for edge computing systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all managers
        self.node_manager = EdgeNodeManager()
        self.placement_optimizer = EdgePlacementOptimizer()
        self.cache_manager = EdgeCacheManager()
        self.performance_optimizer = RegionalPerformanceOptimizer()
        self.failover_manager = EdgeFailoverManager()
        self.monitoring_system = EdgeMonitoringSystem()
        
        # Test configuration
        self.test_deployment_id = "test_deployment_001"
        self.test_nodes = []
        self.test_results = {}
        
        self.logger.info("Edge Computing Integration Test initialized")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite"""
        
        self.logger.info("Starting comprehensive edge computing integration tests")
        start_time = time.time()
        
        test_results = {
            "start_time": start_time,
            "test_suite": "Edge Computing Integration",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": {}
        }
        
        # Test sequence
        test_sequence = [
            ("Node Manager", self._test_edge_node_management),
            ("Placement Optimizer", self._test_placement_optimization),
            ("Cache Manager", self._test_edge_caching),
            ("Performance Optimizer", self._test_performance_optimization),
            ("Failover Manager", self._test_failover_management),
            ("Monitoring System", self._test_monitoring_system),
            ("System Integration", self._test_system_integration),
            ("Load Testing", self._test_load_scenarios),
            ("Failover Scenarios", self._test_failover_scenarios)
        ]
        
        for test_name, test_func in test_sequence:
            try:
                self.logger.info(f"Running test: {test_name}")
                test_result = await test_func()
                
                test_results["test_details"][test_name] = {
                    "status": "passed" if test_result["success"] else "failed",
                    "duration": test_result.get("duration", 0),
                    "metrics": test_result.get("metrics", {}),
                    "details": test_result.get("details", "")
                }
                
                test_results["tests_run"] += 1
                if test_result["success"]:
                    test_results["tests_passed"] += 1
                else:
                    test_results["tests_failed"] += 1
                    
            except Exception as e:
                self.logger.error(f"Test {test_name} failed with exception: {e}")
                test_results["test_details"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                test_results["tests_run"] += 1
                test_results["tests_failed"] += 1
        
        # Calculate overall results
        test_results["end_time"] = time.time()
        test_results["total_duration"] = test_results["end_time"] - start_time
        test_results["success_rate"] = (test_results["tests_passed"] / test_results["tests_run"]) * 100 if test_results["tests_run"] > 0 else 0
        test_results["overall_status"] = "PASSED" if test_results["tests_failed"] == 0 else "FAILED"
        
        self.logger.info(f"Integration tests completed: {test_results['success_rate']:.1f}% success rate")
        
        return test_results
    
    async def _test_edge_node_management(self) -> Dict[str, Any]:
        """Test edge node management functionality"""
        
        test_start = time.time()
        
        try:
            # Create test node specifications
            test_nodes = [
                EdgeNodeSpec(
                    node_id="test_nyse_001",
                    region=TradingRegion.NYSE_MAHWAH,
                    node_type=EdgeNodeType.ULTRA_EDGE,
                    cpu_cores=16,
                    memory_gb=64,
                    target_latency_us=50.0,
                    max_orders_per_second=100000
                ),
                EdgeNodeSpec(
                    node_id="test_nasdaq_001", 
                    region=TradingRegion.NASDAQ_CARTERET,
                    node_type=EdgeNodeType.HIGH_PERFORMANCE,
                    cpu_cores=8,
                    memory_gb=32,
                    target_latency_us=200.0,
                    max_orders_per_second=50000
                ),
                EdgeNodeSpec(
                    node_id="test_cloud_001",
                    region=TradingRegion.US_EAST_1,
                    node_type=EdgeNodeType.STANDARD_EDGE,
                    cpu_cores=4,
                    memory_gb=16,
                    target_latency_us=1000.0,
                    max_orders_per_second=10000
                )
            ]
            
            # Create deployment configuration
            deployment_config = EdgeDeploymentConfig(
                deployment_id=self.test_deployment_id,
                nodes=test_nodes,
                deployment_type="rolling",
                max_unavailable=1,
                health_check_interval_seconds=5
            )
            
            # Deploy nodes
            deployment_result = await self.node_manager.deploy_edge_nodes(deployment_config)
            
            # Start monitoring
            await self.node_manager.start_monitoring()
            
            # Wait for deployment to complete
            await asyncio.sleep(5)
            
            # Verify deployment status
            status = await self.node_manager.get_edge_deployment_status()
            
            # Validate results
            success = (
                deployment_result["deployment_status"] in ["success", "partial_success"] and
                status["deployment_summary"]["total_nodes"] == 3 and
                status["deployment_summary"]["active_nodes"] >= 2
            )
            
            self.test_nodes = test_nodes
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "nodes_deployed": status["deployment_summary"]["total_nodes"],
                    "nodes_active": status["deployment_summary"]["active_nodes"],
                    "deployment_time": deployment_result.get("duration_seconds", 0)
                },
                "details": f"Deployed {len(test_nodes)} nodes, {status['deployment_summary']['active_nodes']} active"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_placement_optimization(self) -> Dict[str, Any]:
        """Test edge placement optimization"""
        
        test_start = time.time()
        
        try:
            # Add trading activity patterns
            trading_activities = [
                TradingActivityPattern(
                    activity_id="hft_nyse",
                    activity_type=TradingActivityType.HIGH_FREQUENCY,
                    primary_markets=["NYSE", "NASDAQ"],
                    max_latency_us=100.0,
                    peak_volume_per_second=50000.0,
                    business_priority=10
                ),
                TradingActivityPattern(
                    activity_id="algo_europe",
                    activity_type=TradingActivityType.ALGORITHMIC,
                    primary_markets=["LSE"],
                    max_latency_us=500.0,
                    peak_volume_per_second=20000.0,
                    business_priority=8
                ),
                TradingActivityPattern(
                    activity_id="execution_asia",
                    activity_type=TradingActivityType.EXECUTION,
                    primary_markets=["TSE", "HKEX"],
                    max_latency_us=1000.0,
                    peak_volume_per_second=10000.0,
                    business_priority=6
                )
            ]
            
            for activity in trading_activities:
                self.placement_optimizer.add_trading_activity_pattern(activity)
            
            # Test different optimization strategies
            strategies_to_test = [
                PlacementStrategy.LATENCY_OPTIMIZED,
                PlacementStrategy.COST_OPTIMIZED,
                PlacementStrategy.BALANCED
            ]
            
            optimization_results = []
            
            for strategy in strategies_to_test:
                result = await self.placement_optimizer.optimize_edge_placement(
                    strategy=strategy,
                    max_nodes=5,
                    budget_constraint_usd=50000.0
                )
                optimization_results.append(result)
            
            # Validate results
            success = all(
                len(result.recommended_placements) > 0 and
                result.optimization_time_seconds < 60.0
                for result in optimization_results
            )
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "strategies_tested": len(strategies_to_test),
                    "avg_optimization_time": sum(r.optimization_time_seconds for r in optimization_results) / len(optimization_results),
                    "total_placements": sum(len(r.recommended_placements) for r in optimization_results),
                    "avg_projected_latency": sum(r.projected_average_latency_us for r in optimization_results) / len(optimization_results)
                },
                "details": f"Tested {len(strategies_to_test)} optimization strategies successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_edge_caching(self) -> Dict[str, Any]:
        """Test edge caching functionality"""
        
        test_start = time.time()
        
        try:
            # Create test caches with different configurations
            cache_configs = [
                CacheConfiguration(
                    cache_id="market_data_cache",
                    node_id="test_nyse_001",
                    region="us_east",
                    max_memory_mb=512,
                    replication_factor=3,
                    cache_strategy=CacheStrategy.WRITE_THROUGH,
                    consistency_level=ConsistencyLevel.STRONG_CONSISTENCY
                ),
                CacheConfiguration(
                    cache_id="analytics_cache",
                    node_id="test_cloud_001",
                    region="us_east",
                    max_memory_mb=256,
                    replication_factor=2,
                    cache_strategy=CacheStrategy.WRITE_BEHIND,
                    consistency_level=ConsistencyLevel.EVENTUAL_CONSISTENCY
                )
            ]
            
            # Create caches
            for config in cache_configs:
                success = await self.cache_manager.create_cache(config)
                if not success:
                    raise Exception(f"Failed to create cache {config.cache_id}")
            
            # Start monitoring
            await self.cache_manager.start_monitoring()
            
            # Test cache operations
            test_data = {
                "AAPL": {"price": 150.25, "volume": 1000000, "timestamp": time.time()},
                "GOOGL": {"price": 2800.50, "volume": 500000, "timestamp": time.time()},
                "MSFT": {"price": 300.75, "volume": 750000, "timestamp": time.time()}
            }
            
            # Set cache items
            cache_operations = 0
            successful_sets = 0
            successful_gets = 0
            
            for symbol, data in test_data.items():
                for config in cache_configs:
                    # Set operation
                    success = await self.cache_manager.set(
                        config.cache_id, 
                        f"market_data_{symbol}",
                        data,
                        DataCategory.MARKET_DATA
                    )
                    cache_operations += 1
                    if success:
                        successful_sets += 1
                    
                    # Get operation
                    retrieved = await self.cache_manager.get(
                        config.cache_id,
                        f"market_data_{symbol}"
                    )
                    cache_operations += 1
                    if retrieved is not None:
                        successful_gets += 1
            
            # Wait for replication
            await asyncio.sleep(2)
            
            # Check cache status
            cache_status = await self.cache_manager.get_cache_status()
            
            success = (
                successful_sets > 0 and
                successful_gets > 0 and
                cache_status["global_stats"]["total_caches"] == len(cache_configs)
            )
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "caches_created": len(cache_configs),
                    "cache_operations": cache_operations,
                    "successful_sets": successful_sets,
                    "successful_gets": successful_gets,
                    "hit_rate": (successful_gets / (successful_sets + successful_gets)) * 100 if (successful_sets + successful_gets) > 0 else 0
                },
                "details": f"Created {len(cache_configs)} caches, performed {cache_operations} operations"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_performance_optimization(self) -> Dict[str, Any]:
        """Test regional performance optimization"""
        
        test_start = time.time()
        
        try:
            # Create test performance profiles
            test_profiles = [
                RegionalPerformanceProfile(
                    region_id="us_east_test",
                    region_name="US East Test Region",
                    latitude=40.7589,
                    longitude=-73.9851,
                    timezone="America/New_York",
                    primary_markets=["NYSE", "NASDAQ"],
                    performance_tier=PerformanceTier.ULTRA_PERFORMANCE,
                    target_latency_us=100.0,
                    current_latency_us=150.0,
                    target_throughput_ops=50000.0,
                    current_throughput_ops=40000.0,
                    allocated_cpu_cores=16,
                    allocated_memory_gb=64,
                    current_cpu_utilization=60.0,
                    current_memory_utilization=50.0,
                    current_bandwidth_utilization=40.0,
                    peak_trading_hours=[(9, 16)],
                    market_session_patterns={}
                ),
                RegionalPerformanceProfile(
                    region_id="eu_west_test",
                    region_name="EU West Test Region",
                    latitude=51.5074,
                    longitude=-0.1278,
                    timezone="Europe/London",
                    primary_markets=["LSE"],
                    performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                    target_latency_us=300.0,
                    current_latency_us=400.0,
                    target_throughput_ops=25000.0,
                    current_throughput_ops=20000.0,
                    allocated_cpu_cores=8,
                    allocated_memory_gb=32,
                    current_cpu_utilization=70.0,
                    current_memory_utilization=65.0,
                    current_bandwidth_utilization=55.0,
                    peak_trading_hours=[(8, 16)],
                    market_session_patterns={}
                )
            ]
            
            # Add profiles
            for profile in test_profiles:
                await self.performance_optimizer.add_regional_profile(profile)
            
            # Start monitoring
            await self.performance_optimizer.start_performance_monitoring()
            
            # Wait for baseline data collection
            await asyncio.sleep(3)
            
            # Test performance optimization for each region
            optimization_results = []
            
            for profile in test_profiles:
                # Create baseline
                baseline = await self.performance_optimizer.create_performance_baseline(profile.region_id)
                
                # Optimize performance
                result = await self.performance_optimizer.optimize_regional_performance(
                    region_id=profile.region_id,
                    objective=OptimizationObjective.BALANCE_PERFORMANCE,
                    target_improvement_percent=20.0
                )
                
                optimization_results.append(result)
            
            # Get optimization summary
            summary = await self.performance_optimizer.get_optimization_summary()
            
            success = (
                len(optimization_results) == len(test_profiles) and
                all(result.recommendations_generated > 0 for result in optimization_results) and
                summary["global_optimization_summary"]["total_regions_optimized"] >= len(test_profiles)
            )
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "regions_tested": len(test_profiles),
                    "optimizations_completed": len(optimization_results),
                    "total_recommendations": sum(r.recommendations_generated for r in optimization_results),
                    "avg_latency_improvement": sum(r.improvement_percentage.get("avg_latency_us", 0) for r in optimization_results) / len(optimization_results)
                },
                "details": f"Optimized {len(test_profiles)} regions with performance improvements"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_failover_management(self) -> Dict[str, Any]:
        """Test edge failover management"""
        
        test_start = time.time()
        
        try:
            # Configure failover
            failover_config = FailoverConfiguration(
                config_id="test_failover_config",
                deployment_name=self.test_deployment_id,
                health_check_interval_seconds=2,
                consecutive_failures_threshold=2,
                max_latency_us=5000.0,
                failover_timeout_seconds=10
            )
            
            await self.failover_manager.configure_failover(failover_config)
            
            # Register test nodes
            for node in self.test_nodes:
                await self.failover_manager.register_edge_node(
                    node_id=node.node_id,
                    deployment_name=self.test_deployment_id
                )
            
            # Start monitoring
            await self.failover_manager.start_monitoring()
            
            # Wait for health checks
            await asyncio.sleep(5)
            
            # Test manual failover
            if self.test_nodes:
                test_node_id = self.test_nodes[0].node_id
                
                await self.failover_manager.manual_failover(
                    node_id=test_node_id,
                    strategy=FailoverStrategy.GRACEFUL,
                    reason="Integration test failover"
                )
                
                # Wait for failover completion
                await asyncio.sleep(3)
            
            # Get failover status
            failover_status = await self.failover_manager.get_failover_status()
            
            success = (
                failover_status["monitoring_active"] and
                failover_status["node_summary"]["total_nodes"] >= len(self.test_nodes) and
                len(failover_status["recent_events"]) > 0
            )
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "nodes_registered": failover_status["node_summary"]["total_nodes"],
                    "active_nodes": failover_status["node_summary"]["active_nodes"],
                    "failed_nodes": failover_status["node_summary"]["failed_nodes"],
                    "recent_events": len(failover_status["recent_events"])
                },
                "details": "Failover system configured and tested successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_monitoring_system(self) -> Dict[str, Any]:
        """Test edge monitoring system"""
        
        test_start = time.time()
        
        try:
            # Start monitoring
            await self.monitoring_system.start_monitoring()
            
            # Add custom alert rules
            test_alert_rules = [
                AlertRule(
                    rule_id="test_latency_alert",
                    rule_name="Test High Latency Alert",
                    metric_name="edge_latency_microseconds",
                    condition="greater_than",
                    threshold=1000.0,
                    severity=AlertSeverity.WARNING,
                    description="Test alert for high latency"
                ),
                AlertRule(
                    rule_id="test_throughput_alert", 
                    rule_name="Test Low Throughput Alert",
                    metric_name="edge_throughput_ops_per_second",
                    condition="less_than",
                    threshold=5000.0,
                    severity=AlertSeverity.WARNING,
                    description="Test alert for low throughput"
                )
            ]
            
            for rule in test_alert_rules:
                self.monitoring_system.add_alert_rule(rule)
            
            # Wait for metric collection
            await asyncio.sleep(15)
            
            # Test metric collection
            latency_metrics = self.monitoring_system.get_metric_values(
                "edge_latency_microseconds",
                time.time() - 300  # Last 5 minutes
            )
            
            throughput_metrics = self.monitoring_system.get_metric_values(
                "edge_throughput_ops_per_second", 
                time.time() - 300
            )
            
            # Get monitoring status
            monitoring_status = self.monitoring_system.get_monitoring_status()
            
            # Get dashboard data
            dashboard_data = self.monitoring_system.get_dashboard_data("edge_overview", 1)
            
            success = (
                monitoring_status["monitoring_active"] and
                monitoring_status["metrics_summary"]["total_metrics"] > 0 and
                len(latency_metrics) > 0 and
                len(throughput_metrics) > 0 and
                "panels" in dashboard_data
            )
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "total_metrics": monitoring_status["metrics_summary"]["total_metrics"],
                    "alert_rules": monitoring_status["alerting_summary"]["total_alert_rules"],
                    "latency_data_points": len(latency_metrics),
                    "throughput_data_points": len(throughput_metrics),
                    "dashboards": monitoring_status["dashboards_summary"]["total_dashboards"]
                },
                "details": "Monitoring system collecting metrics and generating alerts"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_system_integration(self) -> Dict[str, Any]:
        """Test integration between all systems"""
        
        test_start = time.time()
        
        try:
            # Test cross-system data flow
            integration_tests = []
            
            # 1. Node deployment -> Performance monitoring
            node_status = await self.node_manager.get_edge_deployment_status()
            perf_summary = await self.performance_optimizer.get_optimization_summary()
            
            integration_tests.append({
                "test": "node_to_performance",
                "success": node_status["deployment_summary"]["total_nodes"] > 0 and 
                         perf_summary.get("global_optimization_summary", {}).get("total_regions_optimized", 0) > 0
            })
            
            # 2. Cache operations -> Monitoring metrics
            cache_status = await self.cache_manager.get_cache_status()
            monitoring_status = self.monitoring_system.get_monitoring_status()
            
            integration_tests.append({
                "test": "cache_to_monitoring",
                "success": cache_status["global_stats"]["total_caches"] > 0 and
                         monitoring_status["monitoring_active"]
            })
            
            # 3. Performance optimization -> Failover triggers
            failover_status = await self.failover_manager.get_failover_status()
            
            integration_tests.append({
                "test": "performance_to_failover",
                "success": failover_status["monitoring_active"] and
                         failover_status["node_summary"]["total_nodes"] > 0
            })
            
            # 4. Monitoring -> Alerting pipeline
            alert_count = monitoring_status["alerting_summary"]["total_alert_rules"]
            
            integration_tests.append({
                "test": "monitoring_to_alerting",
                "success": alert_count > 0
            })
            
            successful_integrations = sum(1 for test in integration_tests if test["success"])
            success = successful_integrations == len(integration_tests)
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "integration_tests": len(integration_tests),
                    "successful_integrations": successful_integrations,
                    "integration_success_rate": (successful_integrations / len(integration_tests)) * 100
                },
                "details": f"System integration: {successful_integrations}/{len(integration_tests)} tests passed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_load_scenarios(self) -> Dict[str, Any]:
        """Test system under load scenarios"""
        
        test_start = time.time()
        
        try:
            # Simulate high-load scenarios
            load_tests = []
            
            # 1. High cache operations load
            cache_ops_start = time.time()
            cache_operations = 0
            successful_ops = 0
            
            # Perform 100 rapid cache operations
            for i in range(100):
                success = await self.cache_manager.set(
                    "market_data_cache",
                    f"load_test_key_{i}",
                    {"value": i, "timestamp": time.time()},
                    DataCategory.MARKET_DATA
                )
                cache_operations += 1
                if success:
                    successful_ops += 1
                
                # Brief pause to avoid overwhelming
                if i % 10 == 0:
                    await asyncio.sleep(0.01)
            
            cache_ops_duration = time.time() - cache_ops_start
            cache_load_success = successful_ops > 80  # 80% success rate
            
            load_tests.append({
                "test": "cache_load",
                "success": cache_load_success,
                "ops_per_second": cache_operations / cache_ops_duration,
                "success_rate": (successful_ops / cache_operations) * 100
            })
            
            # 2. Multiple concurrent optimizations
            optimization_start = time.time()
            optimization_tasks = []
            
            # Run 3 concurrent placement optimizations
            for i in range(3):
                task = asyncio.create_task(
                    self.placement_optimizer.optimize_edge_placement(
                        strategy=PlacementStrategy.BALANCED,
                        max_nodes=3
                    )
                )
                optimization_tasks.append(task)
            
            # Wait for all optimizations to complete
            optimization_results = await asyncio.gather(*optimization_tasks, return_exceptions=True)
            optimization_duration = time.time() - optimization_start
            
            successful_optimizations = sum(
                1 for result in optimization_results 
                if not isinstance(result, Exception)
            )
            
            optimization_load_success = successful_optimizations >= 2
            
            load_tests.append({
                "test": "concurrent_optimization",
                "success": optimization_load_success,
                "concurrent_tasks": len(optimization_tasks),
                "successful_tasks": successful_optimizations,
                "duration": optimization_duration
            })
            
            # 3. Rapid metric collection
            metric_start = time.time()
            metric_collections = 0
            
            for _ in range(50):
                self.monitoring_system.record_metric(
                    "load_test_metric",
                    time.time() % 1000,  # Varying value
                    labels={"test": "load_scenario"}
                )
                metric_collections += 1
            
            metric_duration = time.time() - metric_start
            metrics_per_second = metric_collections / metric_duration
            
            load_tests.append({
                "test": "metric_collection_load",
                "success": metrics_per_second > 100,  # > 100 metrics/sec
                "metrics_per_second": metrics_per_second,
                "total_metrics": metric_collections
            })
            
            successful_load_tests = sum(1 for test in load_tests if test["success"])
            success = successful_load_tests >= len(load_tests) * 0.8  # 80% success threshold
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "load_tests": len(load_tests),
                    "successful_load_tests": successful_load_tests,
                    "load_success_rate": (successful_load_tests / len(load_tests)) * 100,
                    "max_cache_ops_per_sec": max([test.get("ops_per_second", 0) for test in load_tests]),
                    "max_metrics_per_sec": max([test.get("metrics_per_second", 0) for test in load_tests])
                },
                "details": f"Load testing: {successful_load_tests}/{len(load_tests)} scenarios passed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def _test_failover_scenarios(self) -> Dict[str, Any]:
        """Test various failover scenarios"""
        
        test_start = time.time()
        
        try:
            failover_tests = []
            
            # 1. Test graceful failover
            if self.test_nodes and len(self.test_nodes) > 1:
                graceful_start = time.time()
                
                await self.failover_manager.manual_failover(
                    node_id=self.test_nodes[0].node_id,
                    strategy=FailoverStrategy.GRACEFUL,
                    reason="Graceful failover test"
                )
                
                # Wait for failover completion
                await asyncio.sleep(3)
                
                graceful_duration = time.time() - graceful_start
                graceful_success = graceful_duration < 10.0  # Should complete in < 10 seconds
                
                failover_tests.append({
                    "test": "graceful_failover",
                    "success": graceful_success,
                    "duration": graceful_duration
                })
            
            # 2. Test immediate failover
            if self.test_nodes and len(self.test_nodes) > 1:
                immediate_start = time.time()
                
                await self.failover_manager.manual_failover(
                    node_id=self.test_nodes[1].node_id,
                    strategy=FailoverStrategy.IMMEDIATE,
                    reason="Immediate failover test"
                )
                
                await asyncio.sleep(2)
                
                immediate_duration = time.time() - immediate_start
                immediate_success = immediate_duration < 5.0  # Should complete in < 5 seconds
                
                failover_tests.append({
                    "test": "immediate_failover",
                    "success": immediate_success,
                    "duration": immediate_duration
                })
            
            # 3. Test failover status and recovery
            failover_status = await self.failover_manager.get_failover_status()
            
            status_success = (
                len(failover_status["recent_events"]) >= len(failover_tests) and
                failover_status["monitoring_active"]
            )
            
            failover_tests.append({
                "test": "failover_status",
                "success": status_success,
                "recent_events": len(failover_status["recent_events"])
            })
            
            successful_failover_tests = sum(1 for test in failover_tests if test["success"])
            success = successful_failover_tests >= len(failover_tests) * 0.8
            
            return {
                "success": success,
                "duration": time.time() - test_start,
                "metrics": {
                    "failover_tests": len(failover_tests),
                    "successful_failover_tests": successful_failover_tests,
                    "avg_failover_time": sum([test.get("duration", 0) for test in failover_tests]) / max(len(failover_tests), 1),
                    "failover_success_rate": (successful_failover_tests / len(failover_tests)) * 100 if failover_tests else 0
                },
                "details": f"Failover scenarios: {successful_failover_tests}/{len(failover_tests)} tests passed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
    
    async def cleanup_test_resources(self):
        """Cleanup test resources after testing"""
        
        try:
            # Stop all monitoring systems
            self.node_manager.stop_monitoring()
            self.cache_manager.stop_monitoring()
            self.performance_optimizer.stop_monitoring()
            self.failover_manager.stop_monitoring()
            self.monitoring_system.stop_monitoring()
            
            self.logger.info("Test resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up test resources: {e}")


async def main():
    """Main test execution function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test instance
    integration_test = EdgeComputingIntegrationTest()
    
    try:
        print("🚀 Starting Edge Computing Integration Tests")
        print("=" * 60)
        
        # Run comprehensive tests
        results = await integration_test.run_comprehensive_tests()
        
        # Display results
        print(f"\n📊 Test Results Summary")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Tests Run: {results['tests_run']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        
        print(f"\n📋 Individual Test Results:")
        for test_name, test_result in results['test_details'].items():
            status_icon = "✅" if test_result['status'] == 'passed' else "❌"
            duration = test_result.get('duration', 0)
            print(f"  {status_icon} {test_name}: {test_result['status']} ({duration:.2f}s)")
            
            if 'metrics' in test_result:
                metrics = test_result['metrics']
                print(f"     Metrics: {metrics}")
            
            if 'error' in test_result:
                print(f"     Error: {test_result['error']}")
        
        # Final status
        if results['overall_status'] == 'PASSED':
            print(f"\n🎉 All edge computing integration tests PASSED!")
            print(f"✅ Edge Computing System is ready for production deployment")
        else:
            print(f"\n⚠️  Some integration tests FAILED")
            print(f"❌ Please review failed tests before production deployment")
        
    except Exception as e:
        print(f"\n💥 Integration test execution failed: {e}")
    
    finally:
        # Cleanup
        await integration_test.cleanup_test_resources()
        print(f"\n🧹 Test cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
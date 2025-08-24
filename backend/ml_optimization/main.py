"""
ML Optimization Main Integration Module

This module provides the main integration point for all ML-powered
optimization components, orchestrating the complete system for
Phase 5 enhancement of the Nautilus trading platform.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
import signal
import sys

from .ml_autoscaler import MLAutoScaler
from .predictive_allocator import PredictiveResourceAllocator, AllocationStrategy
from .market_optimizer import MarketConditionOptimizer
from .training_pipeline import MLTrainingPipeline, ModelType, TrainingMode
from .k8s_integration import K8sMLIntegrator
from .performance_monitor import MLPerformanceMonitor


class MLOptimizationOrchestrator:
    """
    Main orchestrator for ML-powered optimization system.
    
    This class coordinates all ML optimization components and provides
    a unified interface for managing the complete system.
    """
    
    def __init__(
        self, 
        namespace: str = "nautilus-trading",
        redis_url: str = "redis://localhost:6379",
        enable_kubernetes: bool = True,
        enable_training: bool = True,
        enable_monitoring: bool = True
    ):
        self.namespace = namespace
        self.redis_url = redis_url
        self.logger = logging.getLogger(__name__)
        
        # Component initialization flags
        self.enable_kubernetes = enable_kubernetes
        self.enable_training = enable_training
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.ml_autoscaler = MLAutoScaler(namespace, redis_url)
        self.resource_allocator = PredictiveResourceAllocator(redis_url)
        self.market_optimizer = MarketConditionOptimizer(redis_url)
        
        if enable_training:
            self.training_pipeline = MLTrainingPipeline(redis_url)
        else:
            self.training_pipeline = None
            
        if enable_kubernetes:
            self.k8s_integrator = K8sMLIntegrator(namespace, redis_url)
        else:
            self.k8s_integrator = None
            
        if enable_monitoring:
            self.performance_monitor = MLPerformanceMonitor(redis_url)
        else:
            self.performance_monitor = None
        
        # System state
        self.is_running = False
        self.optimization_cycle_count = 0
        self.last_optimization_time = None
        
        # Configuration
        self.optimization_interval = 300  # 5 minutes
        self.health_check_interval = 60   # 1 minute
        self.managed_services = [
            "nautilus-market-data",
            "nautilus-strategy-engine",
            "nautilus-risk-engine",
            "nautilus-order-manager",
            "nautilus-position-keeper"
        ]
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the ML optimization system"""
        try:
            self.logger.info("Initializing ML Optimization System...")
            
            initialization_results = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "components": {},
                "warnings": [],
                "errors": []
            }
            
            # Initialize core components
            components = [
                ("ML Auto-Scaler", self.ml_autoscaler),
                ("Resource Allocator", self.resource_allocator),
                ("Market Optimizer", self.market_optimizer)
            ]
            
            if self.training_pipeline:
                components.append(("Training Pipeline", self.training_pipeline))
            
            if self.k8s_integrator:
                components.append(("Kubernetes Integrator", self.k8s_integrator))
            
            if self.performance_monitor:
                components.append(("Performance Monitor", self.performance_monitor))
            
            # Test component initialization
            for component_name, component in components:
                try:
                    # Basic health check for each component
                    if hasattr(component, 'health_check'):
                        health_status = await component.health_check()
                    else:
                        health_status = {"status": "initialized", "component": component_name}
                    
                    initialization_results["components"][component_name] = health_status
                    self.logger.info(f"✅ {component_name} initialized successfully")
                    
                except Exception as e:
                    error_msg = f"Failed to initialize {component_name}: {str(e)}"
                    initialization_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            # Schedule initial model training if enabled
            if self.training_pipeline and self.enable_training:
                try:
                    await self._schedule_initial_training()
                    initialization_results["components"]["Initial Training"] = {"status": "scheduled"}
                except Exception as e:
                    warning_msg = f"Could not schedule initial training: {str(e)}"
                    initialization_results["warnings"].append(warning_msg)
                    self.logger.warning(warning_msg)
            
            # Verify Kubernetes connectivity if enabled
            if self.k8s_integrator and self.enable_kubernetes:
                try:
                    resources = await self.k8s_integrator.get_current_resources()
                    initialization_results["components"]["Kubernetes Resources"] = {
                        "status": "connected",
                        "managed_services": len(resources)
                    }
                except Exception as e:
                    warning_msg = f"Kubernetes connectivity issues: {str(e)}"
                    initialization_results["warnings"].append(warning_msg)
                    self.logger.warning(warning_msg)
            
            # Set overall status
            if initialization_results["errors"]:
                initialization_results["status"] = "partial_failure"
            elif initialization_results["warnings"]:
                initialization_results["status"] = "success_with_warnings"
            
            self.logger.info("ML Optimization System initialization completed")
            return initialization_results
            
        except Exception as e:
            self.logger.error(f"Critical error during initialization: {str(e)}")
            return {
                "status": "failure",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _schedule_initial_training(self):
        """Schedule initial training for all model types"""
        model_types = [
            ModelType.LOAD_PREDICTOR,
            ModelType.PATTERN_CLASSIFIER,
            ModelType.VOLATILITY_PREDICTOR,
            ModelType.REGIME_CLASSIFIER
        ]
        
        for model_type in model_types:
            job_id = self.training_pipeline.schedule_training_job(
                model_type,
                TrainingMode.INITIAL_TRAINING,
                priority=7
            )
            self.logger.info(f"Scheduled initial training for {model_type.value}: {job_id}")
    
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a single optimization cycle across all components"""
        try:
            cycle_start_time = datetime.now()
            self.optimization_cycle_count += 1
            
            self.logger.info(f"Starting optimization cycle #{self.optimization_cycle_count}")
            
            cycle_results = {
                "cycle_number": self.optimization_cycle_count,
                "start_time": cycle_start_time.isoformat(),
                "components": {},
                "summary": {},
                "performance_metrics": {},
                "errors": []
            }
            
            # 1. Analyze market conditions
            try:
                market_condition = await self.market_optimizer.analyze_market_condition()
                optimization_settings = await self.market_optimizer.optimize_for_market_condition(market_condition)
                
                cycle_results["components"]["market_analysis"] = {
                    "regime": market_condition.regime.value,
                    "volatility_level": market_condition.volatility_level,
                    "confidence": market_condition.confidence,
                    "optimization_strategy": optimization_settings.strategy.value
                }
                
                self.logger.info(
                    f"Market Analysis: {market_condition.regime.value}, "
                    f"Volatility: {market_condition.volatility_level:.2f}, "
                    f"Strategy: {optimization_settings.strategy.value}"
                )
                
            except Exception as e:
                error_msg = f"Market analysis failed: {str(e)}"
                cycle_results["errors"].append(error_msg)
                self.logger.error(error_msg)
            
            # 2. ML Auto-scaling decisions
            scaling_results = []
            successful_scalings = 0
            
            for service in self.managed_services:
                try:
                    if self.k8s_integrator:
                        result = await self.k8s_integrator.apply_ml_scaling_decision(service)
                    else:
                        # Direct ML scaling without Kubernetes
                        metrics = await self.ml_autoscaler.collect_metrics(service)
                        prediction = await self.ml_autoscaler.predict_scaling_needs(metrics)
                        result = {
                            "service": service,
                            "action": "ml_recommendation",
                            "scaling_decision": prediction.scaling_recommendation.value,
                            "recommended_replicas": prediction.recommended_replicas,
                            "confidence": prediction.confidence
                        }
                    
                    scaling_results.append(result)
                    
                    if result.get("success", True):
                        successful_scalings += 1
                    
                    # Record performance if monitoring enabled
                    if self.performance_monitor and "confidence" in result:
                        # This would typically compare predicted vs actual outcomes
                        # For now, simulate performance recording
                        await self.performance_monitor.record_scaling_outcome(
                            service_name=service,
                            predicted_need=result.get("scaling_decision", "maintain"),
                            actual_outcome=result.get("action", "unknown"),
                            effectiveness_score=result.get("confidence", 0.5)
                        )
                    
                except Exception as e:
                    error_msg = f"Scaling failed for {service}: {str(e)}"
                    cycle_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
                    
                    scaling_results.append({
                        "service": service,
                        "action": "error",
                        "error": str(e)
                    })
            
            cycle_results["components"]["scaling"] = {
                "total_services": len(self.managed_services),
                "successful_scalings": successful_scalings,
                "results": scaling_results
            }
            
            # 3. Resource allocation optimization
            try:
                if self.k8s_integrator:
                    allocation_result = await self.k8s_integrator.apply_resource_allocation_plan()
                else:
                    # Direct resource allocation
                    allocation_plan = await self.resource_allocator.create_allocation_plan(
                        AllocationStrategy.ML_OPTIMIZED
                    )
                    allocation_result = await self.resource_allocator.execute_allocation_plan(allocation_plan)
                
                cycle_results["components"]["resource_allocation"] = allocation_result
                
                self.logger.info(
                    f"Resource Allocation: {allocation_result.get('status', 'unknown')}, "
                    f"Cost Impact: ${allocation_result.get('total_cost_impact', 0):.2f}"
                )
                
            except Exception as e:
                error_msg = f"Resource allocation failed: {str(e)}"
                cycle_results["errors"].append(error_msg)
                self.logger.error(error_msg)
            
            # 4. Performance monitoring and validation
            if self.performance_monitor:
                try:
                    dashboard_data = await self.performance_monitor.get_monitoring_dashboard_data()
                    active_alerts = await self.performance_monitor.get_active_alerts()
                    
                    cycle_results["components"]["monitoring"] = {
                        "total_predictions": dashboard_data.get("overall_statistics", {}).get("total_predictions", 0),
                        "success_rate": dashboard_data.get("overall_statistics", {}).get("success_rate", 0),
                        "active_alerts": len(active_alerts),
                        "critical_alerts": len([a for a in active_alerts if a.severity.value == "critical"])
                    }
                    
                except Exception as e:
                    error_msg = f"Performance monitoring failed: {str(e)}"
                    cycle_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            # Calculate cycle metrics
            cycle_end_time = datetime.now()
            cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()
            
            cycle_results["end_time"] = cycle_end_time.isoformat()
            cycle_results["duration_seconds"] = cycle_duration
            
            # Summary
            cycle_results["summary"] = {
                "status": "success" if not cycle_results["errors"] else "partial_success",
                "services_optimized": successful_scalings,
                "total_errors": len(cycle_results["errors"]),
                "cycle_duration": f"{cycle_duration:.2f}s"
            }
            
            self.last_optimization_time = cycle_end_time
            
            self.logger.info(
                f"Optimization cycle #{self.optimization_cycle_count} completed in {cycle_duration:.2f}s, "
                f"{successful_scalings}/{len(self.managed_services)} services optimized"
            )
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"Critical error in optimization cycle: {str(e)}")
            return {
                "cycle_number": self.optimization_cycle_count,
                "status": "failure",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_continuous_optimization(self):
        """Run continuous optimization loop"""
        self.logger.info("Starting continuous ML optimization...")
        self.is_running = True
        
        try:
            while self.is_running:
                cycle_start = datetime.now()
                
                # Run optimization cycle
                cycle_result = await self.run_optimization_cycle()
                
                # Log cycle summary
                if cycle_result.get("status") == "success":
                    summary = cycle_result.get("summary", {})
                    self.logger.info(
                        f"Cycle completed: {summary.get('services_optimized', 0)} services optimized, "
                        f"{summary.get('cycle_duration', 'N/A')}"
                    )
                else:
                    self.logger.warning(f"Cycle completed with issues: {cycle_result.get('status', 'unknown')}")
                
                # Calculate sleep time
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, self.optimization_interval - cycle_duration)
                
                if sleep_time > 0:
                    self.logger.debug(f"Waiting {sleep_time:.1f}s until next optimization cycle")
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Continuous optimization cancelled")
        except Exception as e:
            self.logger.error(f"Error in continuous optimization: {str(e)}")
        finally:
            self.is_running = False
            self.logger.info("Continuous optimization stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_running": self.is_running,
                "optimization_cycle_count": self.optimization_cycle_count,
                "last_optimization": self.last_optimization_time.isoformat() if self.last_optimization_time else None,
                "components": {},
                "performance_summary": {},
                "errors": []
            }
            
            # Component status
            components = [
                ("ml_autoscaler", self.ml_autoscaler),
                ("resource_allocator", self.resource_allocator),
                ("market_optimizer", self.market_optimizer)
            ]
            
            if self.training_pipeline:
                components.append(("training_pipeline", self.training_pipeline))
            
            if self.k8s_integrator:
                components.append(("k8s_integrator", self.k8s_integrator))
            
            if self.performance_monitor:
                components.append(("performance_monitor", self.performance_monitor))
            
            for component_name, component in components:
                try:
                    if hasattr(component, 'get_status'):
                        component_status = await component.get_status()
                    else:
                        component_status = {"status": "operational"}
                    
                    status["components"][component_name] = component_status
                    
                except Exception as e:
                    status["components"][component_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    status["errors"].append(f"{component_name}: {str(e)}")
            
            # Performance summary
            if self.performance_monitor:
                try:
                    dashboard_data = await self.performance_monitor.get_monitoring_dashboard_data()
                    status["performance_summary"] = dashboard_data.get("overall_statistics", {})
                except Exception as e:
                    status["errors"].append(f"Performance monitoring: {str(e)}")
            
            # Training pipeline status
            if self.training_pipeline:
                try:
                    training_status = await self.training_pipeline.get_training_status()
                    status["training_summary"] = training_status
                except Exception as e:
                    status["errors"].append(f"Training pipeline: {str(e)}")
            
            return status
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Gracefully shutdown the optimization system"""
        self.logger.info("Shutting down ML Optimization System...")
        
        try:
            self.is_running = False
            
            # Give running tasks time to complete
            await asyncio.sleep(2)
            
            # Component-specific shutdowns could be implemented here
            # For now, just log the shutdown
            
            self.logger.info("ML Optimization System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


async def main():
    """Main entry point for ML Optimization System"""
    parser = argparse.ArgumentParser(description="Nautilus ML Optimization System")
    parser.add_argument("--namespace", default="nautilus-trading", help="Kubernetes namespace")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis connection URL")
    parser.add_argument("--no-kubernetes", action="store_true", help="Disable Kubernetes integration")
    parser.add_argument("--no-training", action="store_true", help="Disable ML training pipeline")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable performance monitoring")
    parser.add_argument("--single-cycle", action="store_true", help="Run single optimization cycle and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    # Initialize orchestrator
    orchestrator = MLOptimizationOrchestrator(
        namespace=args.namespace,
        redis_url=args.redis_url,
        enable_kubernetes=not args.no_kubernetes,
        enable_training=not args.no_training,
        enable_monitoring=not args.no_monitoring
    )
    
    # Signal handling for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(orchestrator.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        print("🤖 Nautilus ML Optimization System")
        print("=" * 50)
        
        init_result = await orchestrator.initialize()
        
        if init_result["status"] == "failure":
            logger.error("System initialization failed")
            sys.exit(1)
        
        logger.info(f"System initialized: {init_result['status']}")
        
        if init_result.get("warnings"):
            for warning in init_result["warnings"]:
                logger.warning(warning)
        
        if args.single_cycle:
            # Run single optimization cycle
            print("\n🔄 Running Single Optimization Cycle...")
            
            cycle_result = await orchestrator.run_optimization_cycle()
            
            print(f"\nCycle Result: {cycle_result.get('summary', {}).get('status', 'unknown')}")
            print(f"Duration: {cycle_result.get('duration_seconds', 0):.2f}s")
            print(f"Services Optimized: {cycle_result.get('summary', {}).get('services_optimized', 0)}")
            
            if cycle_result.get("errors"):
                print("Errors:")
                for error in cycle_result["errors"]:
                    print(f"  - {error}")
            
        else:
            # Run continuous optimization
            print("\n🚀 Starting Continuous Optimization...")
            print("Press Ctrl+C to stop")
            
            await orchestrator.run_continuous_optimization()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Critical system error: {str(e)}")
        sys.exit(1)
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
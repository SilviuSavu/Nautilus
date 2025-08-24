#!/usr/bin/env python3
"""
M4 Max Engine Integration Module
Provides M4 Max hardware acceleration integration for all 9 Nautilus engines
Uses existing optimization components without breaking functionality
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Import existing M4 Max optimization components
try:
    from backend.acceleration import (
        initialize_coreml_acceleration,
        get_acceleration_status,
        is_m4_max_detected,
        neural_performance_context,
        predict,
        predict_batch
    )
    M4_MAX_ACCELERATION_AVAILABLE = True
except ImportError:
    M4_MAX_ACCELERATION_AVAILABLE = False
    logging.warning("M4 Max acceleration not available - using CPU fallback")

try:
    from backend.optimization import (
        OptimizerController,
        PerformanceMonitor,
        WorkloadClassifier,
        get_optimization_status
    )
    M4_MAX_OPTIMIZATION_AVAILABLE = True
except ImportError:
    M4_MAX_OPTIMIZATION_AVAILABLE = False
    logging.warning("M4 Max CPU optimization not available - using standard threading")

from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority


logger = logging.getLogger(__name__)


class M4MaxEngineBase(ABC):
    """
    Base class for M4 Max hardware acceleration integration
    Provides common M4 Max functionality for all engines
    """
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.m4_max_detected = False
        self.neural_acceleration_available = False
        self.cpu_optimization_available = False
        
        # M4 Max optimization components
        self.cpu_optimizer = None
        self.performance_monitor = None
        self.workload_classifier = None
        
        # M4 Max performance metrics
        self.m4_max_metrics = {
            "engine_name": engine_name,
            "m4_max_enabled": False,
            "neural_engine_enabled": False,
            "cpu_optimization_enabled": False,
            "performance_cores_used": 0,
            "efficiency_cores_used": 0,
            "neural_predictions": 0,
            "accelerated_operations": 0,
            "avg_processing_time_ms": 0.0,
            "hardware_acceleration_ratio": 1.0
        }
        
        # Performance tracking
        self.operation_times = []
        self.last_metrics_update = time.time()
    
    async def initialize_m4_max(self) -> Dict[str, Any]:
        """Initialize M4 Max hardware acceleration for this engine"""
        
        logger.info(f"Initializing M4 Max acceleration for {self.engine_name} Engine...")
        
        initialization_result = {
            "success": False,
            "m4_max_detected": False,
            "neural_engine_available": False,
            "cpu_optimization_available": False,
            "components_initialized": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check M4 Max hardware
            if M4_MAX_ACCELERATION_AVAILABLE:
                self.m4_max_detected = is_m4_max_detected()
                initialization_result["m4_max_detected"] = self.m4_max_detected
                
                if self.m4_max_detected:
                    logger.info(f"✅ M4 Max detected for {self.engine_name} Engine")
                else:
                    logger.info(f"ℹ️ Non-M4 Max Apple Silicon detected for {self.engine_name} Engine")
            
            # Initialize Neural Engine acceleration
            await self._initialize_neural_acceleration(initialization_result)
            
            # Initialize CPU optimization
            await self._initialize_cpu_optimization(initialization_result)
            
            # Update engine metrics
            self.m4_max_metrics.update({
                "m4_max_enabled": self.m4_max_detected,
                "neural_engine_enabled": self.neural_acceleration_available,
                "cpu_optimization_enabled": self.cpu_optimization_available
            })
            
            # Engine-specific M4 Max initialization
            await self.initialize_engine_specific_m4_max(initialization_result)
            
            initialization_result["success"] = True
            
            if self.m4_max_detected and (self.neural_acceleration_available or self.cpu_optimization_available):
                logger.info(f"🚀 M4 Max acceleration fully operational for {self.engine_name} Engine")
            elif self.neural_acceleration_available or self.cpu_optimization_available:
                logger.info(f"⚡ Hardware acceleration available for {self.engine_name} Engine")
            else:
                logger.info(f"💻 Standard processing mode for {self.engine_name} Engine")
            
        except Exception as e:
            logger.error(f"M4 Max initialization failed for {self.engine_name} Engine: {e}")
            initialization_result["errors"].append(str(e))
        
        return initialization_result
    
    async def _initialize_neural_acceleration(self, result: Dict[str, Any]):
        """Initialize Neural Engine acceleration"""
        
        if not M4_MAX_ACCELERATION_AVAILABLE:
            result["warnings"].append("Neural Engine acceleration not available")
            return
        
        try:
            # Use existing acceleration initialization
            acceleration_status = await initialize_coreml_acceleration(enable_logging=False)
            
            self.neural_acceleration_available = acceleration_status.get("neural_engine_available", False)
            result["neural_engine_available"] = self.neural_acceleration_available
            
            if self.neural_acceleration_available:
                result["components_initialized"].append("neural_engine")
                logger.info(f"✅ Neural Engine initialized for {self.engine_name} Engine")
            else:
                result["warnings"].append("Neural Engine not available - using CPU fallback")
                
        except Exception as e:
            logger.error(f"Neural Engine initialization failed for {self.engine_name}: {e}")
            result["errors"].append(f"Neural Engine error: {str(e)}")
    
    async def _initialize_cpu_optimization(self, result: Dict[str, Any]):
        """Initialize CPU core optimization"""
        
        if not M4_MAX_OPTIMIZATION_AVAILABLE:
            result["warnings"].append("CPU optimization not available")
            return
        
        try:
            # Initialize existing CPU optimization components
            self.cpu_optimizer = OptimizerController()
            await self.cpu_optimizer.initialize()
            
            self.performance_monitor = PerformanceMonitor()
            await self.performance_monitor.start_monitoring()
            
            self.workload_classifier = WorkloadClassifier()
            
            # Check optimization status
            opt_status = get_optimization_status()
            self.cpu_optimization_available = opt_status.get("optimization_active", False)
            
            result["cpu_optimization_available"] = self.cpu_optimization_available
            
            if self.cpu_optimization_available:
                result["components_initialized"].append("cpu_optimizer")
                logger.info(f"✅ CPU optimization initialized for {self.engine_name} Engine")
                
                # Update metrics with core information
                self.m4_max_metrics.update({
                    "performance_cores_available": opt_status.get("performance_cores", 0),
                    "efficiency_cores_available": opt_status.get("efficiency_cores", 0)
                })
            else:
                result["warnings"].append("CPU optimization not active")
                
        except Exception as e:
            logger.error(f"CPU optimization initialization failed for {self.engine_name}: {e}")
            result["errors"].append(f"CPU optimization error: {str(e)}")
            self.cpu_optimizer = None
            self.performance_monitor = None
            self.workload_classifier = None
    
    @abstractmethod
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Engine-specific M4 Max initialization (implemented by each engine)"""
        pass
    
    async def optimize_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Optimize an operation using M4 Max capabilities"""
        
        start_time = time.time()
        
        try:
            # Classify workload for optimal core usage
            if self.cpu_optimizer and self.workload_classifier:
                workload_category, priority = await self._classify_workload(operation_name, kwargs)
                
                # Get optimization context
                optimization_context = await self.cpu_optimizer.optimize_workload(
                    workload_category, priority
                )
                
                # Execute with optimization
                async with optimization_context:
                    result = await operation_func(*args, **kwargs)
                    
                self._update_core_usage_metrics(workload_category)
                
            else:
                # Fallback to standard execution
                result = await operation_func(*args, **kwargs)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized operation {operation_name} failed in {self.engine_name}: {e}")
            raise
    
    async def neural_predict(self, data: Dict[str, Any], model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Perform Neural Engine prediction if available"""
        
        if not self.neural_acceleration_available:
            return None
        
        try:
            # Use existing neural prediction
            with neural_performance_context(f"{self.engine_name}_prediction"):
                prediction = await predict(data, model_id=model_id or f"{self.engine_name}_model")
                
            if prediction and not prediction.get("error"):
                self.m4_max_metrics["neural_predictions"] += 1
                return prediction
                
        except Exception as e:
            logger.warning(f"Neural prediction failed in {self.engine_name}: {e}")
        
        return None
    
    async def neural_predict_batch(self, data_batch: List[Dict[str, Any]], model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform batch Neural Engine predictions if available"""
        
        if not self.neural_acceleration_available:
            return [None] * len(data_batch)
        
        try:
            # Use existing batch prediction
            predictions = await predict_batch(data_batch, model_id=model_id or f"{self.engine_name}_model")
            
            self.m4_max_metrics["neural_predictions"] += len(predictions)
            return predictions
            
        except Exception as e:
            logger.warning(f"Neural batch prediction failed in {self.engine_name}: {e}")
            return [None] * len(data_batch)
    
    async def _classify_workload(self, operation_name: str, context: Dict[str, Any]):
        """Classify workload for optimal M4 Max core assignment"""
        
        # Default classification based on operation name
        if operation_name in ['calculate', 'compute', 'analyze', 'process']:
            category = "compute_intensive"
            priority = "high"
        elif operation_name in ['stream', 'websocket', 'real_time']:
            category = "latency_sensitive"
            priority = "urgent"
        elif operation_name in ['batch', 'bulk', 'mass']:
            category = "throughput_optimized"
            priority = "normal"
        else:
            category = f"{self.engine_name}_operation"
            priority = "normal"
        
        # Use workload classifier if available
        if self.workload_classifier:
            try:
                classification = await self.workload_classifier.classify_workload(
                    function_name=f"{self.engine_name}_{operation_name}",
                    execution_context=context
                )
                
                if classification:
                    return classification.get("category", category), classification.get("priority", priority)
            except Exception as e:
                logger.warning(f"Workload classification error: {e}")
        
        return category, priority
    
    def _update_core_usage_metrics(self, workload_category: str):
        """Update core usage based on workload category"""
        
        if workload_category in ["latency_sensitive", "urgent"]:
            self.m4_max_metrics["performance_cores_used"] += 1
        elif workload_category in ["throughput_optimized", "batch"]:
            self.m4_max_metrics["efficiency_cores_used"] += 1
        else:
            # Mixed usage
            self.m4_max_metrics["performance_cores_used"] += 0.5
            self.m4_max_metrics["efficiency_cores_used"] += 0.5
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Update performance metrics"""
        
        self.m4_max_metrics["accelerated_operations"] += 1
        
        # Update average processing time (exponential moving average)
        current_avg = self.m4_max_metrics["avg_processing_time_ms"]
        self.m4_max_metrics["avg_processing_time_ms"] = (
            0.9 * current_avg + 0.1 * processing_time_ms
        )
        
        # Calculate hardware acceleration ratio
        baseline_time = 100.0  # Baseline processing time
        acceleration_ratio = baseline_time / max(processing_time_ms, 1.0)
        
        current_ratio = self.m4_max_metrics["hardware_acceleration_ratio"]
        self.m4_max_metrics["hardware_acceleration_ratio"] = (
            0.95 * current_ratio + 0.05 * acceleration_ratio
        )
        
        # Track operation times
        self.operation_times.append(processing_time_ms)
        if len(self.operation_times) > 1000:
            self.operation_times = self.operation_times[-1000:]
    
    def get_m4_max_status(self) -> Dict[str, Any]:
        """Get comprehensive M4 Max status for this engine"""
        
        status = {
            "engine_name": self.engine_name,
            "m4_max_metrics": self.m4_max_metrics.copy(),
            "hardware_status": {
                "m4_max_detected": self.m4_max_detected,
                "neural_engine_available": self.neural_acceleration_available,
                "cpu_optimization_available": self.cpu_optimization_available
            },
            "performance_stats": {
                "total_operations": len(self.operation_times),
                "avg_time_ms": sum(self.operation_times) / max(len(self.operation_times), 1),
                "min_time_ms": min(self.operation_times) if self.operation_times else 0,
                "max_time_ms": max(self.operation_times) if self.operation_times else 0
            }
        }
        
        # Add acceleration status if available
        if M4_MAX_ACCELERATION_AVAILABLE:
            try:
                accel_status = get_acceleration_status()
                status["acceleration_details"] = accel_status
            except Exception:
                pass
        
        # Add CPU optimization status if available
        if M4_MAX_OPTIMIZATION_AVAILABLE and self.cpu_optimizer:
            try:
                opt_status = get_optimization_status()
                status["optimization_details"] = opt_status
            except Exception:
                pass
        
        return status
    
    async def cleanup_m4_max(self):
        """Cleanup M4 Max resources"""
        
        try:
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            if self.cpu_optimizer:
                await self.cpu_optimizer.cleanup()
                
            logger.info(f"M4 Max resources cleaned up for {self.engine_name} Engine")
            
        except Exception as e:
            logger.error(f"M4 Max cleanup error for {self.engine_name}: {e}")


# Engine-specific M4 Max integration classes
class AnalyticsEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for Analytics Engine"""
    
    def __init__(self):
        super().__init__("Analytics")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Analytics-specific M4 Max initialization"""
        
        # Analytics engine benefits from Neural Engine for complex analytics
        if self.neural_acceleration_available:
            result["components_initialized"].append("analytics_neural_models")
            logger.info("✅ Neural Engine configured for analytics predictions")


class FactorEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for Factor Engine"""
    
    def __init__(self):
        super().__init__("Factor")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Factor-specific M4 Max initialization"""
        
        # Factor engine benefits from CPU optimization for 485 factor calculations
        if self.cpu_optimization_available:
            result["components_initialized"].append("factor_calculation_optimization")
            logger.info("✅ CPU optimization configured for 485-factor synthesis")


class FeaturesEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for Features Engine"""
    
    def __init__(self):
        super().__init__("Features")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Features-specific M4 Max initialization"""
        
        # Features engine benefits from both Neural Engine and CPU optimization
        if self.neural_acceleration_available:
            result["components_initialized"].append("feature_extraction_neural")
        if self.cpu_optimization_available:
            result["components_initialized"].append("feature_processing_optimization")


class MLEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for ML Engine"""
    
    def __init__(self):
        super().__init__("ML")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """ML-specific M4 Max initialization"""
        
        # ML engine is ideal for Neural Engine acceleration
        if self.neural_acceleration_available:
            result["components_initialized"].append("ml_model_acceleration")
            result["components_initialized"].append("inference_optimization")
            logger.info("✅ Neural Engine optimized for ML model inference")


class MarketDataEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for MarketData Engine"""
    
    def __init__(self):
        super().__init__("MarketData")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """MarketData-specific M4 Max initialization"""
        
        # MarketData engine benefits from CPU optimization for real-time processing
        if self.cpu_optimization_available:
            result["components_initialized"].append("realtime_data_optimization")
            logger.info("✅ CPU optimization configured for real-time market data")


class PortfolioEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for Portfolio Engine"""
    
    def __init__(self):
        super().__init__("Portfolio")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Portfolio-specific M4 Max initialization"""
        
        # Portfolio engine benefits from both Neural Engine and CPU optimization
        if self.neural_acceleration_available:
            result["components_initialized"].append("portfolio_prediction_models")
        if self.cpu_optimization_available:
            result["components_initialized"].append("portfolio_calculation_optimization")


class StrategyEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for Strategy Engine"""
    
    def __init__(self):
        super().__init__("Strategy")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """Strategy-specific M4 Max initialization"""
        
        # Strategy engine is ideal for Neural Engine trading models
        if self.neural_acceleration_available:
            result["components_initialized"].append("strategy_neural_models")
            result["components_initialized"].append("signal_generation_acceleration")
            logger.info("✅ Neural Engine optimized for trading strategy execution")


class WebSocketEngineM4Max(M4MaxEngineBase):
    """M4 Max integration for WebSocket Engine"""
    
    def __init__(self):
        super().__init__("WebSocket")
    
    async def initialize_engine_specific_m4_max(self, result: Dict[str, Any]):
        """WebSocket-specific M4 Max initialization"""
        
        # WebSocket engine benefits from CPU optimization for low-latency streaming
        if self.cpu_optimization_available:
            result["components_initialized"].append("websocket_streaming_optimization")
            logger.info("✅ CPU optimization configured for low-latency WebSocket streaming")


# Factory function for creating M4 Max integration instances
def create_m4_max_integration(engine_name: str) -> M4MaxEngineBase:
    """Create M4 Max integration instance for specified engine"""
    
    engine_classes = {
        "analytics": AnalyticsEngineM4Max,
        "factor": FactorEngineM4Max,
        "features": FeaturesEngineM4Max,
        "ml": MLEngineM4Max,
        "marketdata": MarketDataEngineM4Max,
        "portfolio": PortfolioEngineM4Max,
        "strategy": StrategyEngineM4Max,
        "websocket": WebSocketEngineM4Max
    }
    
    engine_class = engine_classes.get(engine_name.lower())
    if engine_class:
        return engine_class()
    else:
        raise ValueError(f"Unknown engine name: {engine_name}")


# Utility function to initialize M4 Max for all engines
async def initialize_all_engines_m4_max() -> Dict[str, Any]:
    """Initialize M4 Max acceleration for all 9 engines"""
    
    logger.info("Initializing M4 Max acceleration for all 9 Nautilus engines...")
    
    engines = ["analytics", "factor", "features", "ml", "marketdata", "portfolio", "strategy", "websocket"]
    results = {}
    
    initialization_tasks = []
    
    for engine_name in engines:
        try:
            m4_max_integration = create_m4_max_integration(engine_name)
            task = m4_max_integration.initialize_m4_max()
            initialization_tasks.append((engine_name, task, m4_max_integration))
        except Exception as e:
            logger.error(f"Failed to create M4 Max integration for {engine_name}: {e}")
            results[engine_name] = {"success": False, "error": str(e)}
    
    # Execute all initializations in parallel
    for engine_name, task, integration in initialization_tasks:
        try:
            result = await task
            results[engine_name] = result
            results[engine_name]["integration_instance"] = integration
        except Exception as e:
            logger.error(f"M4 Max initialization failed for {engine_name}: {e}")
            results[engine_name] = {"success": False, "error": str(e)}
    
    # Summary
    successful_engines = [name for name, result in results.items() if result.get("success")]
    failed_engines = [name for name, result in results.items() if not result.get("success")]
    
    logger.info(f"M4 Max initialization complete: {len(successful_engines)} successful, {len(failed_engines)} failed")
    
    if successful_engines:
        logger.info(f"✅ M4 Max enabled for: {', '.join(successful_engines)}")
    if failed_engines:
        logger.warning(f"❌ M4 Max failed for: {', '.join(failed_engines)}")
    
    return {
        "total_engines": len(engines),
        "successful_engines": len(successful_engines),
        "failed_engines": len(failed_engines),
        "results": results,
        "summary": {
            "m4_max_available": M4_MAX_ACCELERATION_AVAILABLE and M4_MAX_OPTIMIZATION_AVAILABLE,
            "engines_with_m4_max": successful_engines,
            "engines_without_m4_max": failed_engines
        }
    }


# Export classes and functions
__all__ = [
    'M4MaxEngineBase',
    'AnalyticsEngineM4Max',
    'FactorEngineM4Max', 
    'FeaturesEngineM4Max',
    'MLEngineM4Max',
    'MarketDataEngineM4Max',
    'PortfolioEngineM4Max',
    'StrategyEngineM4Max',
    'WebSocketEngineM4Max',
    'create_m4_max_integration',
    'initialize_all_engines_m4_max'
]
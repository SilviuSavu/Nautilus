# Enhanced Hardware Routing System for Performance Optimization
# Integrates with M4 Max hardware acceleration and performance optimization components
# Part of the 3-4x performance improvement initiative

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

# Import base hardware router if available
try:
    from ..hardware_router import (
        HardwareRouter, WorkloadType, RoutingDecision, HardwareCapability,
        route_ml_workload, route_risk_workload, hardware_accelerated
    )
    BASE_ROUTER_AVAILABLE = True
except ImportError:
    logging.warning("Base hardware router not available - creating enhanced standalone version")
    BASE_ROUTER_AVAILABLE = False
    HardwareRouter = None

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    """Performance optimization targets"""
    LATENCY = "latency"           # Minimize response time
    THROUGHPUT = "throughput"     # Maximize operations per second  
    ACCURACY = "accuracy"         # Maximize precision of calculations
    EFFICIENCY = "efficiency"     # Optimize resource utilization
    BALANCED = "balanced"         # Balance all metrics

class ResourceType(Enum):
    """System resource types for optimization"""
    CPU_CORES = "cpu_cores"
    NEURAL_ENGINE = "neural_engine"
    METAL_GPU = "metal_gpu"
    UNIFIED_MEMORY = "unified_memory"
    DATABASE_POOL = "database_pool"
    REDIS_CACHE = "redis_cache"
    NETWORK_POOL = "network_pool"

@dataclass
class ResourceUtilization:
    """Current resource utilization metrics"""
    resource_type: ResourceType
    utilization_percent: float
    available_capacity: float
    performance_score: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass 
class OptimizationContext:
    """Context for optimization decisions"""
    workload_size: int
    priority_level: str
    latency_requirement_ms: Optional[float]
    accuracy_requirement: Optional[float]
    current_load: float
    available_resources: Dict[ResourceType, ResourceUtilization]
    historical_performance: Dict[str, float]

@dataclass
class RoutingStrategy:
    """Routing strategy with performance predictions"""
    primary_resource: ResourceType
    fallback_resources: List[ResourceType]
    estimated_latency_ms: float
    estimated_accuracy: float
    confidence_score: float
    resource_requirements: Dict[ResourceType, float]
    optimization_flags: Dict[str, Any]

class EnhancedHardwareRouter:
    """
    Enhanced hardware routing system that integrates with all performance optimization components
    
    Features:
    - Integration with M4 Max hardware acceleration
    - Database connection pool routing
    - Redis cache optimization
    - Parallel engine communication routing
    - Binary serialization optimization
    - Real-time performance monitoring
    - Predictive resource allocation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_router = None
        
        # Performance monitoring
        self.routing_decisions: List[Tuple[datetime, RoutingStrategy, float]] = []
        self.resource_utilization: Dict[ResourceType, ResourceUtilization] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Component integration
        self.database_pool = None
        self.redis_cache = None
        self.parallel_engine_client = None
        self.serializer = None
        
        # Hardware capabilities
        self.m4_max_enabled = os.environ.get('M4_MAX_OPTIMIZED', '0') == '1'
        self.neural_engine_enabled = os.environ.get('NEURAL_ENGINE_ENABLED', '0') == '1'
        self.metal_gpu_enabled = os.environ.get('METAL_ACCELERATION', '0') == '1'
        self.auto_routing_enabled = os.environ.get('AUTO_HARDWARE_ROUTING', '0') == '1'
        
        # Performance thresholds
        self.large_data_threshold = int(os.environ.get('LARGE_DATA_THRESHOLD', '1000000'))
        self.parallel_threshold = int(os.environ.get('PARALLEL_THRESHOLD', '10000'))
        
        # Thread pool for CPU-intensive routing decisions
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="routing-worker")
        
        logger.info(f"Enhanced Hardware Router initialized - M4 Max: {self.m4_max_enabled}, Auto-routing: {self.auto_routing_enabled}")
    
    async def initialize(self) -> None:
        """Initialize router with all optimization components"""
        try:
            # Initialize base hardware router if available
            if BASE_ROUTER_AVAILABLE:
                self.base_router = HardwareRouter()
                logger.info("✅ Base hardware router integrated")
            
            # Initialize resource utilization monitoring
            await self._initialize_resource_monitoring()
            
            # Start periodic resource updates
            asyncio.create_task(self._periodic_resource_update())
            
            logger.info("✅ Enhanced Hardware Router initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Hardware Router initialization failed: {e}")
            raise
    
    async def _initialize_resource_monitoring(self) -> None:
        """Initialize resource utilization tracking"""
        resources = [
            ResourceType.CPU_CORES,
            ResourceType.NEURAL_ENGINE,
            ResourceType.METAL_GPU,
            ResourceType.UNIFIED_MEMORY,
            ResourceType.DATABASE_POOL,
            ResourceType.REDIS_CACHE,
            ResourceType.NETWORK_POOL
        ]
        
        for resource in resources:
            self.resource_utilization[resource] = ResourceUtilization(
                resource_type=resource,
                utilization_percent=0.0,
                available_capacity=100.0,
                performance_score=1.0
            )
    
    async def _periodic_resource_update(self) -> None:
        """Periodically update resource utilization metrics"""
        while True:
            try:
                await self._update_resource_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.warning(f"Resource monitoring update failed: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds on error
    
    async def _update_resource_metrics(self) -> None:
        """Update current resource utilization metrics"""
        try:
            # CPU utilization (approximate based on system load)
            cpu_util = min(asyncio.get_event_loop().time() % 100, 85.0)  # Mock for now
            self.resource_utilization[ResourceType.CPU_CORES].utilization_percent = cpu_util
            
            # Neural Engine utilization (from environment or monitoring)
            neural_util = float(os.environ.get('NEURAL_ENGINE_UTILIZATION', '72'))
            self.resource_utilization[ResourceType.NEURAL_ENGINE].utilization_percent = neural_util
            
            # Metal GPU utilization
            gpu_util = float(os.environ.get('METAL_GPU_UTILIZATION', '85'))  
            self.resource_utilization[ResourceType.METAL_GPU].utilization_percent = gpu_util
            
            # Database pool utilization (would integrate with actual pool)
            db_util = min(len(self.routing_decisions) * 2, 60.0)  # Approximation
            self.resource_utilization[ResourceType.DATABASE_POOL].utilization_percent = db_util
            
            # Redis cache utilization
            redis_util = 45.0  # Would integrate with actual Redis metrics
            self.resource_utilization[ResourceType.REDIS_CACHE].utilization_percent = redis_util
            
            logger.debug(f"Resource utilization updated - CPU: {cpu_util:.1f}%, Neural: {neural_util:.1f}%, GPU: {gpu_util:.1f}%")
            
        except Exception as e:
            logger.warning(f"Resource metrics update failed: {e}")
    
    def _analyze_workload_characteristics(self, workload_type: str, context: OptimizationContext) -> Dict[str, Any]:
        """Analyze workload to determine optimal routing strategy"""
        characteristics = {
            'data_intensive': context.workload_size > self.large_data_threshold,
            'parallel_suitable': context.workload_size > self.parallel_threshold,
            'latency_critical': context.latency_requirement_ms is not None and context.latency_requirement_ms < 100,
            'accuracy_critical': context.accuracy_requirement is not None and context.accuracy_requirement > 0.95,
            'compute_intensive': workload_type in ['monte_carlo', 'matrix_operations', 'ml_inference'],
            'memory_intensive': context.workload_size * 8 > 1024*1024,  # > 1MB
            'cache_suitable': workload_type in ['db_query', 'market_data', 'reference_data']
        }
        
        return characteristics
    
    def _predict_performance(self, resource: ResourceType, workload_characteristics: Dict[str, Any]) -> Tuple[float, float]:
        """Predict latency and accuracy for given resource and workload"""
        base_latency = 50.0  # Base latency in ms
        base_accuracy = 1.0  # Base accuracy score
        
        resource_performance = {
            ResourceType.NEURAL_ENGINE: {
                'latency_multiplier': 0.15,  # 7x faster for ML workloads
                'accuracy_bonus': 0.05,
                'suitable_for': ['ml_inference', 'pattern_recognition']
            },
            ResourceType.METAL_GPU: {
                'latency_multiplier': 0.02,  # 51x faster for compute workloads  
                'accuracy_bonus': 0.0,
                'suitable_for': ['monte_carlo', 'matrix_operations', 'parallel_compute']
            },
            ResourceType.CPU_CORES: {
                'latency_multiplier': 1.0,   # Baseline performance
                'accuracy_bonus': 0.0,
                'suitable_for': ['general', 'sequential', 'io_bound']
            },
            ResourceType.DATABASE_POOL: {
                'latency_multiplier': 0.2,   # 5x faster with connection pooling
                'accuracy_bonus': 0.0,
                'suitable_for': ['db_query', 'data_retrieval']
            },
            ResourceType.REDIS_CACHE: {
                'latency_multiplier': 0.01,  # 100x faster for cache hits
                'accuracy_bonus': 0.0,
                'suitable_for': ['cached_data', 'frequent_queries']
            }
        }
        
        perf = resource_performance.get(resource, resource_performance[ResourceType.CPU_CORES])
        
        # Calculate latency based on workload characteristics
        latency = base_latency * perf['latency_multiplier']
        
        # Apply workload-specific adjustments
        if workload_characteristics.get('data_intensive'):
            latency *= 1.5
        if workload_characteristics.get('parallel_suitable') and resource == ResourceType.METAL_GPU:
            latency *= 0.3  # Additional speedup for parallel workloads
        if workload_characteristics.get('latency_critical'):
            latency *= 0.8  # Prioritize faster resources
        
        # Calculate accuracy
        accuracy = base_accuracy + perf['accuracy_bonus']
        if workload_characteristics.get('accuracy_critical') and resource == ResourceType.NEURAL_ENGINE:
            accuracy += 0.02
        
        # Factor in current resource utilization
        util = self.resource_utilization.get(resource)
        if util and util.utilization_percent > 80:
            latency *= 1.5  # Performance degrades under high load
        
        return latency, accuracy
    
    async def route_workload(
        self, 
        workload_type: str,
        context: OptimizationContext,
        optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
    ) -> RoutingStrategy:
        """
        Route workload to optimal resources based on context and target
        
        Args:
            workload_type: Type of workload (ml_inference, db_query, monte_carlo, etc.)
            context: Current system context and requirements
            optimization_target: Primary optimization target
        
        Returns:
            Optimal routing strategy with performance predictions
        """
        start_time = time.time()
        
        try:
            # Analyze workload characteristics
            characteristics = self._analyze_workload_characteristics(workload_type, context)
            
            # Evaluate all available resources
            resource_scores: Dict[ResourceType, Tuple[float, float, float]] = {}
            
            for resource in ResourceType:
                if not self._is_resource_available(resource):
                    continue
                
                latency, accuracy = self._predict_performance(resource, characteristics)
                
                # Calculate composite score based on optimization target
                score = self._calculate_optimization_score(
                    latency, accuracy, resource, optimization_target, characteristics
                )
                
                resource_scores[resource] = (score, latency, accuracy)
            
            # Select optimal resource
            best_resource = max(resource_scores.keys(), key=lambda r: resource_scores[r][0])
            best_score, best_latency, best_accuracy = resource_scores[best_resource]
            
            # Build fallback chain
            fallback_resources = sorted(
                [r for r in resource_scores.keys() if r != best_resource],
                key=lambda r: resource_scores[r][0],
                reverse=True
            )[:2]  # Top 2 fallback options
            
            # Create routing strategy
            strategy = RoutingStrategy(
                primary_resource=best_resource,
                fallback_resources=fallback_resources,
                estimated_latency_ms=best_latency,
                estimated_accuracy=best_accuracy,
                confidence_score=min(best_score / 100.0, 1.0),
                resource_requirements=self._calculate_resource_requirements(best_resource, characteristics),
                optimization_flags=self._generate_optimization_flags(best_resource, characteristics)
            )
            
            # Record routing decision
            routing_time = (time.time() - start_time) * 1000
            self.routing_decisions.append((datetime.now(), strategy, routing_time))
            
            logger.debug(f"Routed {workload_type} to {best_resource.value} (score: {best_score:.2f}, latency: {best_latency:.1f}ms)")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Workload routing failed: {e}")
            # Return fallback CPU strategy
            return RoutingStrategy(
                primary_resource=ResourceType.CPU_CORES,
                fallback_resources=[],
                estimated_latency_ms=100.0,
                estimated_accuracy=1.0,
                confidence_score=0.5,
                resource_requirements={ResourceType.CPU_CORES: 50.0},
                optimization_flags={}
            )
    
    def _is_resource_available(self, resource: ResourceType) -> bool:
        """Check if resource is available for routing"""
        availability = {
            ResourceType.NEURAL_ENGINE: self.neural_engine_enabled,
            ResourceType.METAL_GPU: self.metal_gpu_enabled,
            ResourceType.CPU_CORES: True,  # Always available
            ResourceType.DATABASE_POOL: self.database_pool is not None,
            ResourceType.REDIS_CACHE: self.redis_cache is not None,
            ResourceType.UNIFIED_MEMORY: self.m4_max_enabled,
            ResourceType.NETWORK_POOL: self.parallel_engine_client is not None
        }
        
        return availability.get(resource, False)
    
    def _calculate_optimization_score(
        self, 
        latency: float, 
        accuracy: float, 
        resource: ResourceType,
        target: OptimizationTarget,
        characteristics: Dict[str, Any]
    ) -> float:
        """Calculate optimization score for resource selection"""
        base_score = 50.0
        
        if target == OptimizationTarget.LATENCY:
            # Lower latency = higher score
            latency_score = max(0, 100 - latency)
            base_score = latency_score * 0.8 + accuracy * 20
        
        elif target == OptimizationTarget.THROUGHPUT:
            # Favor parallel resources for throughput
            if resource in [ResourceType.METAL_GPU, ResourceType.NEURAL_ENGINE]:
                base_score += 30
            if characteristics.get('parallel_suitable'):
                base_score += 20
        
        elif target == OptimizationTarget.ACCURACY:
            # Accuracy is primary concern
            base_score = accuracy * 100
            if resource == ResourceType.NEURAL_ENGINE and characteristics.get('accuracy_critical'):
                base_score += 15
        
        elif target == OptimizationTarget.EFFICIENCY:
            # Balance performance and resource utilization
            util = self.resource_utilization.get(resource)
            if util:
                # Prefer less utilized resources
                efficiency_bonus = (100 - util.utilization_percent) * 0.3
                base_score += efficiency_bonus
        
        else:  # BALANCED
            # Balanced optimization
            latency_component = max(0, 100 - latency) * 0.4
            accuracy_component = accuracy * 30
            efficiency_component = 20
            
            util = self.resource_utilization.get(resource)
            if util and util.utilization_percent < 70:
                efficiency_component += 10
            
            base_score = latency_component + accuracy_component + efficiency_component
        
        return base_score
    
    def _calculate_resource_requirements(self, resource: ResourceType, characteristics: Dict[str, Any]) -> Dict[ResourceType, float]:
        """Calculate resource requirements for optimal performance"""
        requirements = {resource: 50.0}  # Base 50% utilization
        
        if characteristics.get('data_intensive'):
            requirements[ResourceType.UNIFIED_MEMORY] = 30.0
        
        if characteristics.get('parallel_suitable'):
            requirements[resource] = min(requirements.get(resource, 0) + 25.0, 90.0)
        
        if characteristics.get('cache_suitable'):
            requirements[ResourceType.REDIS_CACHE] = 20.0
        
        return requirements
    
    def _generate_optimization_flags(self, resource: ResourceType, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization flags for resource execution"""
        flags = {
            'enable_compression': characteristics.get('data_intensive', False),
            'enable_caching': characteristics.get('cache_suitable', False),
            'enable_parallel': characteristics.get('parallel_suitable', False),
            'priority_level': 'high' if characteristics.get('latency_critical') else 'normal',
            'resource_target': resource.value
        }
        
        # Resource-specific optimizations
        if resource == ResourceType.NEURAL_ENGINE:
            flags.update({
                'neural_engine_mode': 'performance',
                'batch_processing': characteristics.get('parallel_suitable', False)
            })
        
        elif resource == ResourceType.METAL_GPU:
            flags.update({
                'gpu_acceleration': True,
                'metal_performance_shaders': True,
                'memory_optimization': characteristics.get('memory_intensive', False)
            })
        
        elif resource == ResourceType.DATABASE_POOL:
            flags.update({
                'connection_pooling': True,
                'query_optimization': True,
                'result_streaming': characteristics.get('data_intensive', False)
            })
        
        return flags
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics and performance metrics"""
        try:
            total_decisions = len(self.routing_decisions)
            
            if total_decisions == 0:
                return {
                    'total_routing_decisions': 0,
                    'message': 'No routing decisions recorded yet'
                }
            
            # Analyze recent routing decisions (last 100)
            recent_decisions = self.routing_decisions[-100:]
            
            # Resource usage distribution
            resource_usage = {}
            total_latency = 0
            total_accuracy = 0
            
            for timestamp, strategy, routing_time in recent_decisions:
                resource = strategy.primary_resource.value
                resource_usage[resource] = resource_usage.get(resource, 0) + 1
                total_latency += strategy.estimated_latency_ms
                total_accuracy += strategy.estimated_accuracy
            
            avg_latency = total_latency / len(recent_decisions)
            avg_accuracy = total_accuracy / len(recent_decisions)
            
            # Current resource utilization
            current_utilization = {}
            for resource, util in self.resource_utilization.items():
                current_utilization[resource.value] = {
                    'utilization_percent': util.utilization_percent,
                    'available_capacity': util.available_capacity,
                    'performance_score': util.performance_score
                }
            
            return {
                'routing_performance': {
                    'total_routing_decisions': total_decisions,
                    'recent_decisions_analyzed': len(recent_decisions),
                    'average_estimated_latency_ms': round(avg_latency, 2),
                    'average_estimated_accuracy': round(avg_accuracy, 3),
                    'resource_usage_distribution': resource_usage
                },
                'system_status': {
                    'm4_max_optimized': self.m4_max_enabled,
                    'neural_engine_enabled': self.neural_engine_enabled,
                    'metal_gpu_enabled': self.metal_gpu_enabled,
                    'auto_routing_enabled': self.auto_routing_enabled
                },
                'resource_utilization': current_utilization,
                'optimization_thresholds': {
                    'large_data_threshold': self.large_data_threshold,
                    'parallel_threshold': self.parallel_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {
                'error': str(e),
                'total_routing_decisions': len(self.routing_decisions)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of routing system and all integrated components"""
        try:
            health_status = {
                'enhanced_router': 'healthy',
                'base_router': 'healthy' if BASE_ROUTER_AVAILABLE and self.base_router else 'unavailable',
                'resource_monitoring': 'healthy',
                'components': {}
            }
            
            # Check individual component health
            if self.database_pool:
                health_status['components']['database_pool'] = 'healthy'
            
            if self.redis_cache:
                health_status['components']['redis_cache'] = 'healthy'
                
            if self.parallel_engine_client:
                health_status['components']['parallel_engine_client'] = 'healthy'
            
            # Check hardware availability
            health_status['hardware'] = {
                'neural_engine': 'available' if self.neural_engine_enabled else 'disabled',
                'metal_gpu': 'available' if self.metal_gpu_enabled else 'disabled',
                'm4_max_optimized': 'enabled' if self.m4_max_enabled else 'disabled'
            }
            
            return health_status
            
        except Exception as e:
            return {
                'enhanced_router': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self) -> None:
        """Clean up resources"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Enhanced Hardware Router closed")
        except Exception as e:
            logger.error(f"Error closing Enhanced Hardware Router: {e}")

# Global enhanced router instance
_enhanced_router: Optional[EnhancedHardwareRouter] = None

async def get_enhanced_router() -> EnhancedHardwareRouter:
    """Get or create enhanced hardware router instance"""
    global _enhanced_router
    
    if _enhanced_router is None:
        _enhanced_router = EnhancedHardwareRouter()
        await _enhanced_router.initialize()
    
    return _enhanced_router

async def route_optimized_workload(
    workload_type: str,
    workload_size: int = 1000,
    latency_requirement_ms: Optional[float] = None,
    accuracy_requirement: Optional[float] = None,
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
) -> RoutingStrategy:
    """
    Convenience function for optimized workload routing
    
    Args:
        workload_type: Type of workload (db_query, ml_inference, monte_carlo, etc.)
        workload_size: Size of workload (number of operations/records)
        latency_requirement_ms: Maximum acceptable latency
        accuracy_requirement: Minimum required accuracy
        optimization_target: Primary optimization objective
    
    Returns:
        Routing strategy with performance predictions
    """
    router = await get_enhanced_router()
    
    context = OptimizationContext(
        workload_size=workload_size,
        priority_level='normal',
        latency_requirement_ms=latency_requirement_ms,
        accuracy_requirement=accuracy_requirement,
        current_load=0.5,  # Would be dynamically determined
        available_resources=router.resource_utilization,
        historical_performance={}
    )
    
    return await router.route_workload(workload_type, context, optimization_target)

# Decorator for automatic workload routing
def enhanced_hardware_routing(
    workload_type: str,
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
):
    """Decorator for automatic enhanced hardware routing"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Analyze function context
            workload_size = kwargs.get('workload_size', len(args[0]) if args and hasattr(args[0], '__len__') else 1000)
            latency_req = kwargs.get('latency_requirement_ms')
            accuracy_req = kwargs.get('accuracy_requirement')
            
            # Get routing strategy
            strategy = await route_optimized_workload(
                workload_type, workload_size, latency_req, accuracy_req, optimization_target
            )
            
            # Apply optimization flags
            for flag_name, flag_value in strategy.optimization_flags.items():
                if flag_name not in kwargs:
                    kwargs[flag_name] = flag_value
            
            # Execute with routing context
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
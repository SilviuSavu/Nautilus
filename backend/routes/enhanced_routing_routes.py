# Enhanced Hardware Routing API Routes
# FastAPI endpoints for the enhanced hardware routing system

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..routing.enhanced_hardware_router import (
    get_enhanced_router, route_optimized_workload,
    OptimizationTarget, ResourceType, OptimizationContext, RoutingStrategy
)

router = APIRouter(prefix="/api/v1/routing", tags=["Enhanced Hardware Routing"])

# Pydantic models for API requests/responses

class WorkloadRequest(BaseModel):
    """Request model for workload routing"""
    workload_type: str = Field(..., description="Type of workload (db_query, ml_inference, monte_carlo, etc.)")
    workload_size: int = Field(default=1000, description="Size of workload (operations/records)")
    latency_requirement_ms: Optional[float] = Field(None, description="Maximum acceptable latency in milliseconds")
    accuracy_requirement: Optional[float] = Field(None, description="Minimum required accuracy (0.0-1.0)")
    optimization_target: str = Field(default="balanced", description="Optimization target: latency, throughput, accuracy, efficiency, balanced")
    priority_level: str = Field(default="normal", description="Priority level: low, normal, high, urgent")
    current_load: float = Field(default=0.5, description="Current system load (0.0-1.0)")

class RoutingResponse(BaseModel):
    """Response model for workload routing"""
    success: bool
    primary_resource: str
    fallback_resources: List[str]
    estimated_latency_ms: float
    estimated_accuracy: float
    confidence_score: float
    resource_requirements: Dict[str, float]
    optimization_flags: Dict[str, Any]
    routing_decision_time_ms: float

class ResourceUtilizationResponse(BaseModel):
    """Response model for resource utilization"""
    resource_type: str
    utilization_percent: float
    available_capacity: float
    performance_score: float
    last_updated: datetime

class RoutingStatisticsResponse(BaseModel):
    """Response model for routing statistics"""
    routing_performance: Dict[str, Any]
    system_status: Dict[str, Any]
    resource_utilization: Dict[str, Any]
    optimization_thresholds: Dict[str, Any]

@router.post("/route-workload", response_model=RoutingResponse)
async def route_workload(request: WorkloadRequest):
    """
    Route workload to optimal hardware resources
    
    Analyzes the workload characteristics and system state to determine
    the optimal routing strategy for maximum performance.
    """
    try:
        start_time = datetime.now()
        
        # Convert optimization target string to enum
        target_mapping = {
            "latency": OptimizationTarget.LATENCY,
            "throughput": OptimizationTarget.THROUGHPUT,
            "accuracy": OptimizationTarget.ACCURACY,
            "efficiency": OptimizationTarget.EFFICIENCY,
            "balanced": OptimizationTarget.BALANCED
        }
        
        optimization_target = target_mapping.get(request.optimization_target.lower(), OptimizationTarget.BALANCED)
        
        # Route the workload
        strategy = await route_optimized_workload(
            workload_type=request.workload_type,
            workload_size=request.workload_size,
            latency_requirement_ms=request.latency_requirement_ms,
            accuracy_requirement=request.accuracy_requirement,
            optimization_target=optimization_target
        )
        
        # Calculate routing decision time
        routing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return RoutingResponse(
            success=True,
            primary_resource=strategy.primary_resource.value,
            fallback_resources=[r.value for r in strategy.fallback_resources],
            estimated_latency_ms=strategy.estimated_latency_ms,
            estimated_accuracy=strategy.estimated_accuracy,
            confidence_score=strategy.confidence_score,
            resource_requirements={k.value if hasattr(k, 'value') else str(k): v for k, v in strategy.resource_requirements.items()},
            optimization_flags=strategy.optimization_flags,
            routing_decision_time_ms=routing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Workload routing failed: {str(e)}"
        )

@router.get("/statistics", response_model=RoutingStatisticsResponse)
async def get_routing_statistics():
    """
    Get comprehensive routing statistics and performance metrics
    
    Returns detailed analytics about routing decisions, resource utilization,
    and system performance over time.
    """
    try:
        router_instance = await get_enhanced_router()
        stats = await router_instance.get_routing_statistics()
        
        return RoutingStatisticsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get routing statistics: {str(e)}"
        )

@router.get("/resource-utilization", response_model=Dict[str, ResourceUtilizationResponse])
async def get_resource_utilization():
    """
    Get current resource utilization for all system resources
    
    Provides real-time view of resource consumption across CPU, GPU,
    Neural Engine, memory, database pools, and caches.
    """
    try:
        router_instance = await get_enhanced_router()
        utilization = router_instance.resource_utilization
        
        response = {}
        for resource_type, util_data in utilization.items():
            response[resource_type.value] = ResourceUtilizationResponse(
                resource_type=resource_type.value,
                utilization_percent=util_data.utilization_percent,
                available_capacity=util_data.available_capacity,
                performance_score=util_data.performance_score,
                last_updated=util_data.last_updated
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resource utilization: {str(e)}"
        )

@router.get("/health")
async def get_routing_health():
    """
    Check health of enhanced routing system
    
    Verifies that all routing components are operational and reports
    the status of hardware acceleration features.
    """
    try:
        router_instance = await get_enhanced_router()
        health_status = await router_instance.health_check()
        
        return {
            "status": "healthy" if health_status.get("enhanced_router") == "healthy" else "unhealthy",
            "details": health_status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/benchmark")
async def run_routing_benchmark(
    workload_types: List[str] = Query(default=["db_query", "ml_inference", "monte_carlo"], description="Workload types to benchmark"),
    iterations: int = Query(default=100, description="Number of iterations per workload"),
    workload_sizes: List[int] = Query(default=[1000, 10000, 100000], description="Workload sizes to test")
):
    """
    Run routing performance benchmark
    
    Tests routing performance across different workload types and sizes
    to validate optimal resource allocation decisions.
    """
    try:
        benchmark_results = {
            "benchmark_config": {
                "workload_types": workload_types,
                "iterations": iterations,
                "workload_sizes": workload_sizes
            },
            "results": {}
        }
        
        for workload_type in workload_types:
            benchmark_results["results"][workload_type] = {}
            
            for workload_size in workload_sizes:
                size_results = {
                    "iterations": iterations,
                    "routing_decisions": [],
                    "average_routing_time_ms": 0.0,
                    "resource_distribution": {}
                }
                
                total_routing_time = 0.0
                
                for i in range(iterations):
                    start_time = datetime.now()
                    
                    strategy = await route_optimized_workload(
                        workload_type=workload_type,
                        workload_size=workload_size,
                        optimization_target=OptimizationTarget.BALANCED
                    )
                    
                    routing_time = (datetime.now() - start_time).total_seconds() * 1000
                    total_routing_time += routing_time
                    
                    # Track resource selection
                    resource = strategy.primary_resource.value
                    size_results["resource_distribution"][resource] = size_results["resource_distribution"].get(resource, 0) + 1
                    
                    # Store decision details (sample every 10th iteration to avoid huge responses)
                    if i % 10 == 0:
                        size_results["routing_decisions"].append({
                            "iteration": i,
                            "primary_resource": resource,
                            "estimated_latency_ms": strategy.estimated_latency_ms,
                            "confidence_score": strategy.confidence_score,
                            "routing_time_ms": routing_time
                        })
                
                size_results["average_routing_time_ms"] = total_routing_time / iterations
                benchmark_results["results"][workload_type][f"size_{workload_size}"] = size_results
        
        return {
            "benchmark_completed": True,
            "total_routing_decisions": len(workload_types) * len(workload_sizes) * iterations,
            "results": benchmark_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Routing benchmark failed: {str(e)}"
        )

@router.get("/optimization-recommendations")
async def get_optimization_recommendations(
    workload_type: str = Query(..., description="Workload type for recommendations"),
    current_performance_ms: float = Query(..., description="Current performance in milliseconds")
):
    """
    Get optimization recommendations for specific workload
    
    Analyzes current performance and suggests optimizations to improve
    latency, throughput, or resource efficiency.
    """
    try:
        # Get current routing strategy
        strategy = await route_optimized_workload(
            workload_type=workload_type,
            optimization_target=OptimizationTarget.BALANCED
        )
        
        recommendations = []
        
        # Compare current vs predicted performance
        if current_performance_ms > strategy.estimated_latency_ms * 1.5:
            recommendations.append({
                "type": "performance_optimization",
                "priority": "high",
                "description": f"Current performance ({current_performance_ms:.1f}ms) significantly slower than optimal ({strategy.estimated_latency_ms:.1f}ms)",
                "action": f"Consider using {strategy.primary_resource.value} for better performance",
                "estimated_improvement": f"{((current_performance_ms - strategy.estimated_latency_ms) / current_performance_ms * 100):.1f}% faster"
            })
        
        # Resource-specific recommendations
        router_instance = await get_enhanced_router()
        
        if strategy.primary_resource == ResourceType.NEURAL_ENGINE:
            neural_util = router_instance.resource_utilization.get(ResourceType.NEURAL_ENGINE)
            if neural_util and neural_util.utilization_percent > 80:
                recommendations.append({
                    "type": "resource_optimization",
                    "priority": "medium",
                    "description": "Neural Engine utilization is high",
                    "action": "Consider batch processing or temporal load balancing",
                    "estimated_improvement": "20-30% better resource efficiency"
                })
        
        if strategy.primary_resource == ResourceType.METAL_GPU:
            gpu_util = router_instance.resource_utilization.get(ResourceType.METAL_GPU)
            if gpu_util and gpu_util.utilization_percent > 85:
                recommendations.append({
                    "type": "scaling_recommendation",
                    "priority": "medium", 
                    "description": "Metal GPU utilization near capacity",
                    "action": "Consider workload scheduling or hybrid CPU+GPU processing",
                    "estimated_improvement": "Maintain performance under high load"
                })
        
        # Cache optimization recommendations
        if workload_type in ["db_query", "market_data", "reference_data"]:
            recommendations.append({
                "type": "caching_optimization",
                "priority": "low",
                "description": "Workload suitable for aggressive caching",
                "action": "Enable Redis caching with HYBRID strategy",
                "estimated_improvement": "50-90% faster for repeated queries"
            })
        
        return {
            "workload_type": workload_type,
            "current_performance_ms": current_performance_ms,
            "optimal_strategy": {
                "primary_resource": strategy.primary_resource.value,
                "estimated_latency_ms": strategy.estimated_latency_ms,
                "confidence_score": strategy.confidence_score
            },
            "recommendations": recommendations,
            "optimization_flags": strategy.optimization_flags
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate optimization recommendations: {str(e)}"
        )
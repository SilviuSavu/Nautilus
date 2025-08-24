"""
Edge Computing API Routes for Nautilus Trading Platform

This module provides REST API endpoints for edge computing management,
deployment, monitoring, and optimization operations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
from datetime import datetime

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
    ReplicationMode, ConsistencyLevel, DataCategory
)
from .regional_performance_optimizer import (
    RegionalPerformanceOptimizer, RegionalPerformanceProfile,
    OptimizationObjective, PerformanceTier
)
from .edge_failover_manager import (
    EdgeFailoverManager, FailoverConfiguration, FailoverStrategy,
    ConsistencyModel, NodeStatus as FailoverNodeStatus
)
from .edge_monitoring_system import (
    EdgeMonitoringSystem, AlertRule, AlertSeverity, MetricType
)


# Pydantic models for API requests/responses
class EdgeNodeSpecRequest(BaseModel):
    node_id: str
    region: str
    node_type: str
    cpu_cores: int = 16
    memory_gb: int = 64
    storage_gb: int = 1000
    network_bandwidth_gbps: int = 25
    target_latency_us: float = 100.0
    max_orders_per_second: int = 100000
    dedicated_network: bool = True
    sr_iov_enabled: bool = True
    cpu_isolation: bool = True


class EdgeDeploymentRequest(BaseModel):
    deployment_id: str
    nodes: List[EdgeNodeSpecRequest]
    deployment_type: str = "rolling"
    max_unavailable: int = 1
    health_check_interval_seconds: int = 5


class CacheConfigurationRequest(BaseModel):
    cache_id: str
    node_id: str
    region: str
    max_memory_mb: int = 1024
    max_items: int = 100000
    replication_factor: int = 3
    cache_strategy: str = "write_behind"
    consistency_level: str = "eventual"


class TradingActivityRequest(BaseModel):
    activity_id: str
    activity_type: str
    primary_markets: List[str]
    max_latency_us: float = 1000.0
    peak_volume_per_second: float = 10000.0
    business_priority: int = 5


class OptimizationRequest(BaseModel):
    strategy: str = "balanced"
    max_nodes: int = 10
    budget_constraint_usd: Optional[float] = None
    latency_constraint_us: Optional[float] = None
    geographic_constraints: Optional[List[str]] = None


class PerformanceProfileRequest(BaseModel):
    region_id: str
    region_name: str
    latitude: float
    longitude: float
    timezone: str
    primary_markets: List[str]
    performance_tier: str
    target_latency_us: float
    target_throughput_ops: float
    allocated_cpu_cores: int = 16
    allocated_memory_gb: int = 64


class AlertRuleRequest(BaseModel):
    rule_id: str
    rule_name: str
    metric_name: str
    condition: str
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"
    description: str = ""


# Initialize managers (singleton pattern)
edge_node_manager = EdgeNodeManager()
placement_optimizer = EdgePlacementOptimizer()
cache_manager = EdgeCacheManager()
performance_optimizer = RegionalPerformanceOptimizer()
failover_manager = EdgeFailoverManager()
monitoring_system = EdgeMonitoringSystem()

# Create API router
router = APIRouter(prefix="/api/v1/edge", tags=["Edge Computing"])

# Logging
logger = logging.getLogger(__name__)


@router.on_event("startup")
async def startup_edge_systems():
    """Initialize edge computing systems on startup"""
    try:
        # Start all monitoring systems
        await edge_node_manager.start_monitoring()
        await cache_manager.start_monitoring()
        await performance_optimizer.start_performance_monitoring()
        await failover_manager.start_monitoring()
        await monitoring_system.start_monitoring()
        
        logger.info("Edge computing systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize edge computing systems: {e}")


@router.on_event("shutdown")
async def shutdown_edge_systems():
    """Cleanup edge computing systems on shutdown"""
    try:
        edge_node_manager.stop_monitoring()
        cache_manager.stop_monitoring()
        performance_optimizer.stop_monitoring()
        failover_manager.stop_monitoring()
        monitoring_system.stop_monitoring()
        
        logger.info("Edge computing systems shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down edge computing systems: {e}")


# Edge Node Management Endpoints
@router.post("/nodes/deploy")
async def deploy_edge_nodes(
    deployment_request: EdgeDeploymentRequest,
    background_tasks: BackgroundTasks
):
    """Deploy edge nodes across regions"""
    try:
        # Convert request to internal model
        node_specs = []
        for node_req in deployment_request.nodes:
            spec = EdgeNodeSpec(
                node_id=node_req.node_id,
                region=TradingRegion(node_req.region),
                node_type=EdgeNodeType(node_req.node_type),
                cpu_cores=node_req.cpu_cores,
                memory_gb=node_req.memory_gb,
                storage_gb=node_req.storage_gb,
                network_bandwidth_gbps=node_req.network_bandwidth_gbps,
                target_latency_us=node_req.target_latency_us,
                max_orders_per_second=node_req.max_orders_per_second,
                dedicated_network=node_req.dedicated_network,
                sr_iov_enabled=node_req.sr_iov_enabled,
                cpu_isolation=node_req.cpu_isolation
            )
            node_specs.append(spec)
        
        deployment_config = EdgeDeploymentConfig(
            deployment_id=deployment_request.deployment_id,
            nodes=node_specs,
            deployment_type=deployment_request.deployment_type,
            max_unavailable=deployment_request.max_unavailable,
            health_check_interval_seconds=deployment_request.health_check_interval_seconds
        )
        
        # Execute deployment in background
        result = await edge_node_manager.deploy_edge_nodes(deployment_config)
        
        return {
            "status": "success",
            "deployment_id": deployment_request.deployment_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Edge node deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/status")
async def get_edge_nodes_status():
    """Get status of all edge nodes"""
    try:
        status = await edge_node_manager.get_edge_deployment_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get edge nodes status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}/health")
async def get_node_health(node_id: str):
    """Get health status of specific edge node"""
    try:
        if node_id not in edge_node_manager.node_metrics:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        metrics = edge_node_manager.node_metrics[node_id]
        return {
            "node_id": node_id,
            "metrics": {
                "latency_us": metrics.avg_latency_us,
                "throughput_ops": metrics.orders_per_second,
                "health_score": metrics.health_score,
                "availability": metrics.availability,
                "timestamp": metrics.timestamp
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Edge Placement Optimization Endpoints
@router.post("/placement/activities")
async def add_trading_activity(activity_request: TradingActivityRequest):
    """Add trading activity pattern for placement optimization"""
    try:
        pattern = TradingActivityPattern(
            activity_id=activity_request.activity_id,
            activity_type=TradingActivityType(activity_request.activity_type),
            primary_markets=activity_request.primary_markets,
            max_latency_us=activity_request.max_latency_us,
            peak_volume_per_second=activity_request.peak_volume_per_second,
            business_priority=activity_request.business_priority
        )
        
        placement_optimizer.add_trading_activity_pattern(pattern)
        
        return {
            "status": "success",
            "message": f"Trading activity pattern {activity_request.activity_id} added"
        }
    except Exception as e:
        logger.error(f"Failed to add trading activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/placement/optimize")
async def optimize_edge_placement(optimization_request: OptimizationRequest):
    """Optimize edge node placement"""
    try:
        strategy = PlacementStrategy(optimization_request.strategy)
        
        result = await placement_optimizer.optimize_edge_placement(
            strategy=strategy,
            max_nodes=optimization_request.max_nodes,
            budget_constraint_usd=optimization_request.budget_constraint_usd,
            latency_constraint_us=optimization_request.latency_constraint_us,
            geographic_constraints=optimization_request.geographic_constraints
        )
        
        # Convert result to serializable format
        return {
            "optimization_id": result.optimization_id,
            "strategy": result.strategy_used.value,
            "recommendations_count": len(result.recommended_placements),
            "projected_cost_monthly": result.projected_total_cost_usd_monthly,
            "projected_latency_us": result.projected_average_latency_us,
            "projected_availability": result.projected_availability,
            "recommended_placements": [
                {
                    "candidate_id": placement.candidate_id,
                    "region": placement.region_name,
                    "provider": placement.cloud_provider,
                    "estimated_cost": placement.estimated_monthly_cost_usd,
                    "overall_score": placement.overall_score
                }
                for placement in result.recommended_placements
            ],
            "alternative_strategies": result.alternative_strategies,
            "optimization_time": result.optimization_time_seconds
        }
        
    except Exception as e:
        logger.error(f"Edge placement optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/placement/recommendations")
async def get_placement_recommendations():
    """Get placement optimization recommendations summary"""
    try:
        summary = placement_optimizer.get_placement_recommendations_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get placement recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Edge Cache Management Endpoints
@router.post("/cache/create")
async def create_edge_cache(cache_request: CacheConfigurationRequest):
    """Create new edge cache"""
    try:
        config = CacheConfiguration(
            cache_id=cache_request.cache_id,
            node_id=cache_request.node_id,
            region=cache_request.region,
            max_memory_mb=cache_request.max_memory_mb,
            max_items=cache_request.max_items,
            replication_factor=cache_request.replication_factor,
            cache_strategy=CacheStrategy(cache_request.cache_strategy),
            consistency_level=ConsistencyLevel(cache_request.consistency_level)
        )
        
        success = await cache_manager.create_cache(config)
        
        if success:
            return {
                "status": "success",
                "cache_id": cache_request.cache_id,
                "message": "Edge cache created successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create cache")
            
    except Exception as e:
        logger.error(f"Failed to create edge cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/{cache_id}")
async def get_cache_item(cache_id: str, key: str):
    """Get item from edge cache"""
    try:
        value = await cache_manager.get(cache_id, key)
        
        if value is not None:
            return {"cache_id": cache_id, "key": key, "value": value, "found": True}
        else:
            return {"cache_id": cache_id, "key": key, "found": False}
            
    except Exception as e:
        logger.error(f"Failed to get cache item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cache/{cache_id}")
async def set_cache_item(
    cache_id: str, 
    key: str, 
    value: Any = Body(...),
    category: str = "market_data",
    ttl_seconds: Optional[int] = None
):
    """Set item in edge cache"""
    try:
        success = await cache_manager.set(
            cache_id=cache_id,
            key=key,
            value=value,
            category=DataCategory(category),
            ttl_seconds=ttl_seconds
        )
        
        if success:
            return {"status": "success", "cache_id": cache_id, "key": key}
        else:
            raise HTTPException(status_code=400, detail="Failed to set cache item")
            
    except Exception as e:
        logger.error(f"Failed to set cache item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/status")
async def get_cache_status():
    """Get comprehensive cache status"""
    try:
        status = await cache_manager.get_cache_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Regional Performance Optimization Endpoints
@router.post("/performance/profiles")
async def add_performance_profile(profile_request: PerformanceProfileRequest):
    """Add regional performance profile"""
    try:
        profile = RegionalPerformanceProfile(
            region_id=profile_request.region_id,
            region_name=profile_request.region_name,
            latitude=profile_request.latitude,
            longitude=profile_request.longitude,
            timezone=profile_request.timezone,
            primary_markets=profile_request.primary_markets,
            performance_tier=PerformanceTier(profile_request.performance_tier),
            target_latency_us=profile_request.target_latency_us,
            current_latency_us=profile_request.target_latency_us * 1.2,  # Initial estimate
            target_throughput_ops=profile_request.target_throughput_ops,
            current_throughput_ops=profile_request.target_throughput_ops * 0.8,  # Initial estimate
            allocated_cpu_cores=profile_request.allocated_cpu_cores,
            allocated_memory_gb=profile_request.allocated_memory_gb,
            current_cpu_utilization=50.0,  # Initial estimate
            current_memory_utilization=60.0,  # Initial estimate
            current_bandwidth_utilization=40.0,  # Initial estimate
            peak_trading_hours=[(9, 16)],  # Default market hours
            market_session_patterns={}
        )
        
        await performance_optimizer.add_regional_profile(profile)
        
        return {
            "status": "success",
            "region_id": profile_request.region_id,
            "message": "Regional performance profile added"
        }
        
    except Exception as e:
        logger.error(f"Failed to add performance profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/optimize/{region_id}")
async def optimize_regional_performance(
    region_id: str,
    objective: str = "balance_performance",
    target_improvement: float = 20.0
):
    """Optimize performance for specific region"""
    try:
        optimization_objective = OptimizationObjective(objective)
        
        result = await performance_optimizer.optimize_regional_performance(
            region_id=region_id,
            objective=optimization_objective,
            target_improvement_percent=target_improvement
        )
        
        return {
            "optimization_id": result.optimization_id,
            "region_id": result.region_id,
            "objective": result.optimization_objective.value,
            "recommendations_count": result.recommendations_generated,
            "high_priority_count": result.high_priority_recommendations,
            "projected_improvements": result.improvement_percentage,
            "immediate_actions": len(result.immediate_actions),
            "scheduled_actions": len(result.scheduled_actions),
            "estimated_cost": result.estimated_total_cost
        }
        
    except Exception as e:
        logger.error(f"Regional performance optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary")
async def get_performance_summary():
    """Get optimization summary across all regions"""
    try:
        summary = await performance_optimizer.get_optimization_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Edge Failover Management Endpoints
@router.post("/failover/configure")
async def configure_failover(
    config_id: str,
    deployment_name: str,
    health_check_interval: int = 5,
    consecutive_failures_threshold: int = 3,
    max_latency_us: float = 2000.0,
    failover_timeout: int = 30
):
    """Configure failover settings"""
    try:
        config = FailoverConfiguration(
            config_id=config_id,
            deployment_name=deployment_name,
            health_check_interval_seconds=health_check_interval,
            consecutive_failures_threshold=consecutive_failures_threshold,
            max_latency_us=max_latency_us,
            failover_timeout_seconds=failover_timeout,
            consistency_model=ConsistencyModel.EVENTUAL_CONSISTENCY
        )
        
        await failover_manager.configure_failover(config)
        
        return {
            "status": "success",
            "config_id": config_id,
            "message": "Failover configuration applied"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure failover: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/failover/manual/{node_id}")
async def trigger_manual_failover(
    node_id: str,
    strategy: str = "graceful",
    reason: str = "Manual operator intervention"
):
    """Manually trigger failover for specific node"""
    try:
        failover_strategy = FailoverStrategy(strategy)
        
        result = await failover_manager.manual_failover(
            node_id=node_id,
            strategy=failover_strategy,
            reason=reason
        )
        
        return {"status": "success", "message": result}
        
    except Exception as e:
        logger.error(f"Manual failover failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failover/status")
async def get_failover_status():
    """Get comprehensive failover status"""
    try:
        status = await failover_manager.get_failover_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get failover status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Edge Monitoring Endpoints
@router.post("/monitoring/alerts")
async def add_alert_rule(alert_request: AlertRuleRequest):
    """Add custom alert rule"""
    try:
        alert_rule = AlertRule(
            rule_id=alert_request.rule_id,
            rule_name=alert_request.rule_name,
            metric_name=alert_request.metric_name,
            condition=alert_request.condition,
            threshold=alert_request.threshold,
            duration_seconds=alert_request.duration_seconds,
            severity=AlertSeverity(alert_request.severity),
            description=alert_request.description
        )
        
        monitoring_system.add_alert_rule(alert_rule)
        
        return {
            "status": "success",
            "rule_id": alert_request.rule_id,
            "message": "Alert rule added"
        }
        
    except Exception as e:
        logger.error(f"Failed to add alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/metrics/{metric_name}")
async def get_metric_data(
    metric_name: str,
    hours: int = Query(1, description="Time range in hours", ge=1, le=24)
):
    """Get metric data for specified time range"""
    try:
        time_range_seconds = hours * 3600
        end_time = time.time()
        start_time = end_time - time_range_seconds
        
        points = monitoring_system.get_metric_values(metric_name, start_time, end_time)
        
        return {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "data_points": len(points),
            "data": [
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "labels": point.labels
                }
                for point in points
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get metric data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/dashboards/{dashboard_id}")
async def get_dashboard_data(
    dashboard_id: str,
    hours: int = Query(1, description="Time range in hours", ge=1, le=24)
):
    """Get dashboard data for visualization"""
    try:
        data = monitoring_system.get_dashboard_data(dashboard_id, hours)
        return data
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        status = monitoring_system.get_monitoring_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/reports/latest")
async def get_latest_performance_report():
    """Get latest performance report"""
    try:
        report = monitoring_system.get_latest_performance_report()
        
        if report:
            return {
                "report_id": report.report_id,
                "generated_at": report.generated_at,
                "time_range_hours": report.time_range_hours,
                "summary": {
                    "avg_latency_us": report.avg_latency_us,
                    "p99_latency_us": report.p99_latency_us,
                    "error_rate": report.error_rate,
                    "availability": report.availability
                },
                "trends": {
                    "latency": report.latency_trend,
                    "throughput": report.throughput_trend,
                    "error": report.error_trend
                },
                "recommendations": {
                    "optimizations": report.optimization_opportunities,
                    "capacity": report.capacity_recommendations,
                    "alerts": report.alert_recommendations
                }
            }
        else:
            return {"message": "No performance reports available"}
            
    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Global Status Endpoint
@router.get("/status")
async def get_edge_computing_status():
    """Get comprehensive edge computing system status"""
    try:
        return {
            "timestamp": time.time(),
            "systems": {
                "edge_nodes": await edge_node_manager.get_edge_deployment_status(),
                "cache_manager": await cache_manager.get_cache_status(),
                "performance_optimizer": await performance_optimizer.get_optimization_summary(),
                "failover_manager": await failover_manager.get_failover_status(),
                "monitoring_system": monitoring_system.get_monitoring_status()
            },
            "health": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get edge computing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint  
@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "edge-computing",
        "version": "1.0.0"
    }
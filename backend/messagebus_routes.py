"""
MessageBus Optimization API Routes
Provides REST API endpoints for Redis pub/sub connection optimization
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from messagebus_optimization import (
    get_messagebus_optimizer,
    optimize_all_messagebus_connections,
    get_messagebus_status,
    benchmark_messagebus_performance
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/messagebus", tags=["MessageBus Optimization"])


@router.get("/status")
async def messagebus_status() -> Dict[str, Any]:
    """Get comprehensive MessageBus optimization status"""
    try:
        status = await get_messagebus_status()
        return {
            "success": True,
            "messagebus": status,
            "summary": {
                "redis_connected": status.get("redis_connected", False),
                "total_streams": status.get("total_streams", 0),
                "total_engines": 9,
                "optimization_active": status.get("total_streams", 0) > 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to get MessageBus status: {e}")
        raise HTTPException(status_code=500, detail=f"MessageBus status error: {str(e)}")


@router.post("/optimize")
async def optimize_messagebus_connections() -> Dict[str, Any]:
    """Optimize MessageBus Redis pub/sub connections for all engines"""
    try:
        results = await optimize_all_messagebus_connections()
        
        # Count successful optimizations
        successful_optimizations = sum(1 for r in results.values() if r.get("optimized", False))
        total_engines = len(results)
        
        return {
            "success": True,
            "message": f"Optimized MessageBus connections for {successful_optimizations}/{total_engines} engines",
            "optimization_results": results,
            "summary": {
                "engines_optimized": successful_optimizations,
                "total_engines": total_engines,
                "success_rate": f"{(successful_optimizations/total_engines*100):.1f}%" if total_engines > 0 else "0%"
            }
        }
    except Exception as e:
        logger.error(f"Failed to optimize MessageBus connections: {e}")
        raise HTTPException(status_code=500, detail=f"MessageBus optimization error: {str(e)}")


@router.get("/performance/benchmark")
async def benchmark_messagebus() -> Dict[str, Any]:
    """Benchmark MessageBus performance with optimized settings"""
    try:
        benchmark_results = await benchmark_messagebus_performance()
        
        overall_performance = benchmark_results.get("overall_performance", {})
        
        return {
            "success": True,
            "message": f"MessageBus performance benchmark completed - {overall_performance.get('rating', 'unknown')} performance",
            "benchmark": benchmark_results,
            "performance_summary": {
                "overall_rating": overall_performance.get("rating", "unknown"),
                "avg_throughput": f"{overall_performance.get('avg_messages_per_second', 0):.1f} msg/sec",
                "avg_latency": f"{overall_performance.get('avg_latency_ms', 0):.1f}ms",
                "engines_tested": overall_performance.get("engines_optimized", 0)
            }
        }
    except Exception as e:
        logger.error(f"Failed to benchmark MessageBus performance: {e}")
        raise HTTPException(status_code=500, detail=f"MessageBus benchmark error: {str(e)}")


@router.get("/streams")
async def get_stream_details() -> Dict[str, Any]:
    """Get detailed information about all MessageBus streams"""
    try:
        optimizer = await get_messagebus_optimizer()
        status = await optimizer.get_optimization_status()
        
        stream_stats = status.get("stream_statistics", {})
        
        # Format stream details for API response
        stream_details = {}
        for engine_name, stats in stream_stats.items():
            if "error" not in stats:
                stream_details[engine_name] = {
                    "stream_key": stats.get("stream_key"),
                    "message_count": stats.get("length", 0),
                    "consumer_groups": stats.get("consumer_groups", 0),
                    "last_message_id": stats.get("last_id", "0-0"),
                    "memory_usage": {
                        "radix_tree_keys": stats.get("radix_tree_keys", 0),
                        "radix_tree_nodes": stats.get("radix_tree_nodes", 0)
                    },
                    "status": "active" if stats.get("length", 0) > 0 else "idle"
                }
            else:
                stream_details[engine_name] = {
                    "status": "error",
                    "error": stats.get("error")
                }
        
        return {
            "success": True,
            "streams": stream_details,
            "summary": {
                "total_streams": len([s for s in stream_details.values() if s.get("status") != "error"]),
                "active_streams": len([s for s in stream_details.values() if s.get("status") == "active"]),
                "idle_streams": len([s for s in stream_details.values() if s.get("status") == "idle"]),
                "error_streams": len([s for s in stream_details.values() if s.get("status") == "error"])
            }
        }
    except Exception as e:
        logger.error(f"Failed to get stream details: {e}")
        raise HTTPException(status_code=500, detail=f"Stream details error: {str(e)}")


@router.get("/config")
async def get_optimization_config() -> Dict[str, Any]:
    """Get MessageBus optimization configuration for all engines"""
    try:
        optimizer = await get_messagebus_optimizer()
        
        # Extract configuration details
        config_details = {}
        for engine_type, config in optimizer._engine_configs.items():
            config_details[engine_type.value] = {
                "stream_key": config.stream_key,
                "consumer_group": config.consumer_group,
                "consumer_name": config.consumer_name,
                "performance_settings": {
                    "buffer_interval_ms": config.buffer_interval_ms,
                    "max_buffer_size": config.max_buffer_size,
                    "heartbeat_interval_secs": config.heartbeat_interval_secs
                },
                "connection_settings": {
                    "connection_timeout": config.connection_timeout,
                    "max_reconnect_attempts": config.max_reconnect_attempts,
                    "reconnect_base_delay": config.reconnect_base_delay,
                    "reconnect_max_delay": config.reconnect_max_delay
                },
                "message_routing": {
                    "topic_filter": config.topic_filter or [],
                    "priority_topics": config.priority_topics or []
                }
            }
        
        return {
            "success": True,
            "configurations": config_details,
            "summary": {
                "total_engines": len(config_details),
                "optimized_settings": True,
                "redis_host": "redis",
                "redis_port": 6379
            }
        }
    except Exception as e:
        logger.error(f"Failed to get optimization config: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval error: {str(e)}")


@router.post("/reset")
async def reset_messagebus_optimization() -> Dict[str, Any]:
    """Reset MessageBus optimization (recreate streams and consumer groups)"""
    try:
        optimizer = await get_messagebus_optimizer()
        
        # Re-initialize to reset all configurations
        success = await optimizer.initialize()
        
        if success:
            # Re-optimize all connections
            results = await optimizer.optimize_connections()
            successful_resets = sum(1 for r in results.values() if r.get("optimized", False))
            
            return {
                "success": True,
                "message": f"MessageBus optimization reset completed for {successful_resets} engines",
                "reset_results": results
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reset MessageBus optimization")
            
    except Exception as e:
        logger.error(f"Failed to reset MessageBus optimization: {e}")
        raise HTTPException(status_code=500, detail=f"MessageBus reset error: {str(e)}")


@router.get("/health")
async def messagebus_health_check() -> Dict[str, Any]:
    """Health check for MessageBus optimization system"""
    try:
        optimizer = await get_messagebus_optimizer()
        status = await optimizer.get_optimization_status()
        
        redis_connected = status.get("redis_connected", False)
        total_streams = status.get("total_streams", 0)
        
        health_status = "healthy" if redis_connected and total_streams > 0 else "degraded" if redis_connected else "unhealthy"
        
        return {
            "status": health_status,
            "redis_connected": redis_connected,
            "streams_active": total_streams,
            "engines_configured": 9,
            "optimization_active": total_streams > 0,
            "redis_version": status.get("redis_version"),
            "connected_clients": status.get("connected_clients", 0),
            "timestamp": status.get("optimization_timestamp")
        }
    except Exception as e:
        logger.error(f"MessageBus health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "redis_connected": False,
            "streams_active": 0,
            "optimization_active": False
        }
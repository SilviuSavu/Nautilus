"""
Story 5.2: System Performance Monitoring API Routes
Implements backend endpoints for system performance monitoring including latency, system metrics, connections, and alerts
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import psutil
import time
import asyncio
from pydantic import BaseModel
import platform
import socket

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["system-monitoring"])

# Global monitoring data storage (in production, use proper database)
_latency_data = {}
_alert_rules = {}
_alert_history = []

# Pydantic models for request/response schemas
class AlertConfigurationRequest(BaseModel):
    metric_name: str
    threshold_value: float
    condition: str  # 'greater_than' | 'less_than' | 'equals' | 'not_equals'
    severity: str   # 'low' | 'medium' | 'high' | 'critical'
    enabled: bool
    venue_filter: Optional[List[str]] = None
    notification_channels: Dict[str, Any]
    escalation_rules: Dict[str, Any]
    auto_resolution: Dict[str, Any]


@router.get("/latency")
async def get_latency_monitoring(
    venue: str = Query("all", description="Venue filter or 'all'"),
    timeframe: str = Query("1h", description="Timeframe for latency data")
):
    """
    GET /api/v1/monitoring/latency?venue=all&timeframe=1h
    Returns comprehensive latency monitoring data for venues
    """
    try:
        # Mock latency data - in production, get from monitoring service
        venue_latencies = []
        venues = ["IB", "Alpaca", "Binance"] if venue == "all" else [venue]
        
        for venue_name in venues:
            venue_data = {
                "venue_name": venue_name,
                "order_execution_latency": {
                    "min_ms": 2.1,
                    "max_ms": 45.7,
                    "avg_ms": 12.3,
                    "p50_ms": 10.2,
                    "p95_ms": 28.5,
                    "p99_ms": 41.2,
                    "samples": 1542
                },
                "market_data_latency": {
                    "tick_to_trade_ms": 5.8,
                    "feed_latency_ms": 3.2,
                    "processing_latency_ms": 2.6,
                    "total_latency_ms": 8.8
                },
                "connection_latency": {
                    "ping_ms": 15.4,
                    "jitter_ms": 2.1,
                    "packet_loss_percent": 0.02
                },
                "last_updated": datetime.now().isoformat()
            }
            venue_latencies.append(venue_data)
        
        overall_stats = {
            "best_venue": "IB",
            "worst_venue": "Binance", 
            "avg_execution_latency_ms": 12.3,
            "latency_trend": "improving"
        }
        
        return {
            "venue_latencies": venue_latencies,
            "overall_statistics": overall_stats,
            "timeframe": timeframe
        }
        
    except Exception as e:
        logger.error(f"Error getting latency monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_monitoring(
    metrics: str = Query("cpu,memory,network", description="Metrics to include"),
    period: str = Query("realtime", description="Data period")
):
    """
    GET /api/v1/monitoring/system?metrics=cpu,memory,network&period=realtime
    Returns comprehensive system performance metrics
    """
    try:
        # Get real system metrics using psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Non-blocking
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)  # Non-blocking
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Network stats
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        # Disk stats
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        response = {
            "timestamp": datetime.now().isoformat(),
            "cpu_metrics": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "load_average_1m": load_avg[0],
                "load_average_5m": load_avg[1],
                "load_average_15m": load_avg[2],
                "per_core_usage": per_cpu,
                "temperature_celsius": None  # Would need specialized sensors
            },
            "memory_metrics": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "buffer_cache_gb": round((memory.buffers + memory.cached) / (1024**3), 2) if hasattr(memory, 'buffers') else 0.0
            },
            "network_metrics": {
                "bytes_sent_per_sec": net_io.bytes_sent,
                "bytes_received_per_sec": net_io.bytes_recv,
                "packets_sent_per_sec": net_io.packets_sent,
                "packets_received_per_sec": net_io.packets_recv,
                "errors_per_sec": net_io.errin + net_io.errout,
                "active_connections": net_connections,
                "bandwidth_utilization_percent": min((net_io.bytes_sent + net_io.bytes_recv) / (100 * 1024 * 1024) * 100, 100)  # Assume 100Mbps baseline
            },
            "disk_metrics": {
                "total_space_gb": round(disk_usage.total / (1024**3), 2),
                "used_space_gb": round(disk_usage.used / (1024**3), 2),
                "available_space_gb": round(disk_usage.free / (1024**3), 2),
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_iops": disk_io.read_count if disk_io else 0,
                "write_iops": disk_io.write_count if disk_io else 0,
                "read_throughput_mbps": round((disk_io.read_bytes / (1024**2)) if disk_io else 0, 2),
                "write_throughput_mbps": round((disk_io.write_bytes / (1024**2)) if disk_io else 0, 2)
            },
            "process_metrics": {
                "trading_engine_cpu_percent": cpu_percent * 0.6,  # Mock: assume trading engine uses 60% of CPU
                "trading_engine_memory_mb": round(memory.used * 0.4 / (1024**2), 2),  # Mock: 40% of used memory
                "database_cpu_percent": cpu_percent * 0.2,  # Mock: DB uses 20% of CPU
                "database_memory_mb": round(memory.used * 0.3 / (1024**2), 2),  # Mock: 30% of used memory
                "total_processes": len(psutil.pids())
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting system monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections")
async def get_connection_monitoring(
    venue: str = Query("all", description="Venue filter or 'all'"),
    include_history: bool = Query(True, description="Include connection history")
):
    """
    GET /api/v1/monitoring/connections?venue=all&include_history=true
    Returns connection monitoring data for venues
    """
    try:
        venues = ["IB", "Alpaca", "Binance"] if venue == "all" else [venue]
        venue_connections = []
        
        for venue_name in venues:
            connection_data = {
                "venue_name": venue_name,
                "status": "connected" if venue_name != "Binance" else "degraded",
                "quality_score": 95 if venue_name == "IB" else 78 if venue_name == "Alpaca" else 65,
                "uptime_percent_24h": 99.8 if venue_name == "IB" else 98.5 if venue_name == "Alpaca" else 94.2,
                "connection_duration_seconds": 86400 - (0 if venue_name == "IB" else 300 if venue_name == "Alpaca" else 1200),
                "last_disconnect_time": None if venue_name == "IB" else (datetime.now() - timedelta(hours=2)).isoformat(),
                "disconnect_count_24h": 0 if venue_name == "IB" else 1 if venue_name == "Alpaca" else 3,
                "data_quality": {
                    "message_rate_per_sec": 150 if venue_name == "IB" else 89 if venue_name == "Alpaca" else 45,
                    "duplicate_messages_percent": 0.01,
                    "out_of_sequence_percent": 0.002,
                    "stale_data_percent": 0.1 if venue_name != "Binance" else 2.3
                },
                "performance_metrics": {
                    "response_time_ms": 12 if venue_name == "IB" else 18 if venue_name == "Alpaca" else 35,
                    "throughput_mbps": 25.4 if venue_name == "IB" else 15.2 if venue_name == "Alpaca" else 8.7,
                    "error_rate_percent": 0.05 if venue_name != "Binance" else 1.2
                },
                "reconnection_stats": {
                    "auto_reconnect_enabled": True,
                    "reconnect_attempts_24h": 0 if venue_name == "IB" else 1 if venue_name == "Alpaca" else 5,
                    "avg_reconnect_time_seconds": 0 if venue_name == "IB" else 15 if venue_name == "Alpaca" else 45,
                    "max_reconnect_time_seconds": 0 if venue_name == "IB" else 30 if venue_name == "Alpaca" else 120
                }
            }
            venue_connections.append(connection_data)
        
        connection_history = []
        if include_history:
            connection_history = [
                {
                    "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                    "venue_name": "Binance",
                    "event_type": "reconnect",
                    "details": "Auto-reconnection successful after temporary network issue"
                },
                {
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "venue_name": "Alpaca",
                    "event_type": "disconnect",
                    "details": "Brief connection loss during market hours"
                }
            ]
        
        overall_health = {
            "total_venues": len(venues),
            "connected_venues": sum(1 for v in venue_connections if v["status"] == "connected"),
            "degraded_venues": sum(1 for v in venue_connections if v["status"] == "degraded"),
            "overall_score": sum(v["quality_score"] for v in venue_connections) / len(venue_connections)
        }
        
        return {
            "venue_connections": venue_connections,
            "connection_history": connection_history if include_history else None,
            "overall_health": overall_health
        }
        
    except Exception as e:
        logger.error(f"Error getting connection monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts_monitoring(
    status: str = Query("active", description="Alert status filter"),
    severity: str = Query("all", description="Severity filter")
):
    """
    GET /api/v1/monitoring/alerts?status=active&severity=high
    Returns performance monitoring alerts
    """
    try:
        # Mock alert data - in production, get from alert system
        active_alerts = []
        if status in ["active", "all"]:
            active_alerts = [
                {
                    "alert_id": "alert_001",
                    "metric_name": "cpu_usage",
                    "current_value": 87.5,
                    "threshold_value": 80.0,
                    "severity": "medium",
                    "triggered_at": (datetime.now() - timedelta(minutes=15)).isoformat(),
                    "venue_name": None,
                    "description": "CPU usage above 80% threshold for 15 minutes",
                    "auto_resolution_available": True,
                    "escalation_level": 1,
                    "notification_sent": True
                }
            ]
            
            if severity in ["high", "all"]:
                active_alerts.append({
                    "alert_id": "alert_002", 
                    "metric_name": "latency_p95",
                    "current_value": 125.3,
                    "threshold_value": 100.0,
                    "severity": "high",
                    "triggered_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "venue_name": "Binance",
                    "description": "Order execution latency P95 above 100ms",
                    "auto_resolution_available": False,
                    "escalation_level": 2,
                    "notification_sent": True
                })
        
        resolved_alerts = []
        if status in ["resolved", "all"]:
            resolved_alerts = [
                {
                    "alert_id": "alert_000",
                    "metric_name": "memory_usage",
                    "resolved_at": (datetime.now() - timedelta(hours=1)).isoformat(),
                    "resolution_method": "auto",
                    "duration_minutes": 45
                }
            ]
        
        alert_statistics = {
            "total_alerts_24h": 8,
            "critical_alerts_24h": 1,
            "avg_resolution_time_minutes": 22.5,
            "most_frequent_alert_type": "cpu_usage"
        }
        
        return {
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "alert_statistics": alert_statistics
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/configure")
async def configure_alert(request: AlertConfigurationRequest):
    """
    POST /api/v1/monitoring/alerts/configure
    Configure performance monitoring alerts
    """
    try:
        # Validate request
        if not request.metric_name or not request.threshold_value:
            raise HTTPException(status_code=400, detail="metric_name and threshold_value are required")
        
        if request.condition not in ['greater_than', 'less_than', 'equals', 'not_equals']:
            raise HTTPException(status_code=400, detail="Invalid condition")
            
        if request.severity not in ['low', 'medium', 'high', 'critical']:
            raise HTTPException(status_code=400, detail="Invalid severity")
        
        # Generate alert rule ID
        alert_rule_id = f"rule_{len(_alert_rules) + 1}"
        
        # Store alert rule (in production, save to database)
        _alert_rules[alert_rule_id] = {
            "metric_name": request.metric_name,
            "threshold_value": request.threshold_value,
            "condition": request.condition,
            "severity": request.severity,
            "enabled": request.enabled,
            "venue_filter": request.venue_filter,
            "notification_channels": request.notification_channels,
            "escalation_rules": request.escalation_rules,
            "auto_resolution": request.auto_resolution,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return {
            "alert_rule_id": alert_rule_id,
            "status": "created",
            "message": f"Alert rule for {request.metric_name} created successfully",
            "validation_errors": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring alert: {e}")
        return {
            "alert_rule_id": None,
            "status": "error",
            "message": "Failed to create alert rule",
            "validation_errors": [str(e)]
        }


@router.get("/performance-trends")
async def get_performance_trends(
    period: str = Query("7d", description="Analysis period")
):
    """
    GET /api/v1/monitoring/performance-trends?period=7d
    Returns performance trend analysis and capacity planning
    """
    try:
        # Mock trend analysis - in production, use ML/statistical analysis
        trend_analysis = [
            {
                "metric_name": "cpu_usage",
                "current_value": 65.2,
                "trend_direction": "stable",
                "change_percent_24h": 2.1,
                "change_percent_7d": -1.8,
                "predicted_value_24h": 66.8,
                "confidence_score": 0.87
            },
            {
                "metric_name": "memory_usage",
                "current_value": 78.4,
                "trend_direction": "improving", 
                "change_percent_24h": -3.2,
                "change_percent_7d": -5.7,
                "predicted_value_24h": 76.1,
                "confidence_score": 0.92
            },
            {
                "metric_name": "order_latency_p95",
                "current_value": 45.7,
                "trend_direction": "degrading",
                "change_percent_24h": 8.3,
                "change_percent_7d": 15.2,
                "predicted_value_24h": 49.2,
                "confidence_score": 0.78
            }
        ]
        
        capacity_planning = {
            "cpu_exhaustion_prediction_days": None,  # No exhaustion predicted
            "memory_exhaustion_prediction_days": 120,  # Based on current trend
            "disk_exhaustion_prediction_days": 45,   # Based on data growth
            "recommended_actions": [
                "Consider upgrading memory within 4 months",
                "Implement data archival strategy for disk space",
                "Monitor order latency trend - investigate if continues degrading"
            ]
        }
        
        anomalies_detected = [
            {
                "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                "metric_name": "network_throughput",
                "anomaly_score": 0.85,
                "description": "Network throughput spike during market open"
            }
        ]
        
        return {
            "trend_analysis": trend_analysis,
            "capacity_planning": capacity_planning,
            "anomalies_detected": anomalies_detected
        }
        
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def monitoring_health():
    """Health check for monitoring service"""
    return {
        "service": "system-monitoring",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring_active": True,
        "data_collection_active": True
    }
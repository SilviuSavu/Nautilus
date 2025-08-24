"""
M4 Max Monitoring Routes for Nautilus Trading Platform
FastAPI routes for comprehensive M4 Max hardware and performance monitoring.

Endpoints:
- /api/v1/monitoring/m4max/hardware/metrics - M4 Max hardware metrics
- /api/v1/monitoring/containers/metrics - Container performance metrics
- /api/v1/monitoring/trading/metrics - Trading performance metrics
- /api/v1/monitoring/dashboard/summary - Production dashboard summary
- /api/v1/monitoring/alerts - Alert management
- /api/v1/monitoring/health - System health status
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import redis

from .m4max_hardware_monitor import M4MaxHardwareMonitor
from .container_performance_monitor import ContainerPerformanceMonitor
from .trading_performance_monitor import TradingPerformanceMonitor
from .production_monitoring_dashboard import ProductionMonitoringDashboard, AlertSeverity, SystemHealthStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global monitoring instances (initialized in main.py)
m4max_monitor: Optional[M4MaxHardwareMonitor] = None
container_monitor: Optional[ContainerPerformanceMonitor] = None
trading_monitor: Optional[TradingPerformanceMonitor] = None
production_dashboard: Optional[ProductionMonitoringDashboard] = None

# Create router
router = APIRouter(prefix="/api/v1/monitoring", tags=["M4 Max Monitoring"])

# Pydantic models
class M4MaxMetricsResponse(BaseModel):
    """M4 Max hardware metrics response"""
    timestamp: datetime
    cpu_p_cores_usage: float = Field(..., description="P-cores utilization percentage")
    cpu_e_cores_usage: float = Field(..., description="E-cores utilization percentage")
    cpu_frequency_mhz: float = Field(..., description="CPU frequency in MHz")
    unified_memory_usage_gb: float = Field(..., description="Unified memory usage in GB")
    unified_memory_bandwidth_gbps: float = Field(..., description="Memory bandwidth in GB/s")
    gpu_utilization_percent: float = Field(..., description="GPU utilization percentage")
    neural_engine_utilization_percent: float = Field(..., description="Neural Engine utilization")
    neural_engine_tops_used: float = Field(..., description="Neural Engine TOPS used")
    thermal_state: str = Field(..., description="Thermal state")
    power_consumption_watts: float = Field(..., description="Power consumption in watts")

class ContainerMetricsResponse(BaseModel):
    """Container metrics response"""
    timestamp: datetime
    containers: List[Dict[str, Any]] = Field(..., description="Container metrics list")
    engine_health: List[Dict[str, Any]] = Field(..., description="Engine health metrics")
    summary: Dict[str, Any] = Field(..., description="Container summary")

class TradingMetricsResponse(BaseModel):
    """Trading performance metrics response"""
    timestamp: datetime
    overall_score: float = Field(..., description="Overall performance score")
    latency_metrics: List[Dict[str, Any]] = Field(..., description="Latency metrics")
    throughput_metrics: List[Dict[str, Any]] = Field(..., description="Throughput metrics")
    risk_metrics: Dict[str, Any] = Field(..., description="Risk assessment metrics")
    ml_metrics: List[Dict[str, Any]] = Field(..., description="ML inference metrics")

class SystemHealthResponse(BaseModel):
    """System health response"""
    overall_status: str = Field(..., description="Overall system health status")
    performance_score: float = Field(..., description="Performance score (0-100)")
    active_alerts: int = Field(..., description="Number of active alerts")
    critical_alerts: int = Field(..., description="Number of critical alerts")
    warning_alerts: int = Field(..., description="Number of warning alerts")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    optimization_opportunities: List[str] = Field(..., description="Optimization opportunities")
    recommendations: List[str] = Field(..., description="System recommendations")

class AlertResponse(BaseModel):
    """Alert response"""
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False

class NotificationConfigRequest(BaseModel):
    """Notification configuration request"""
    slack_webhook: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: Optional[int] = None
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    pagerduty_key: Optional[str] = None

# Dependency to get monitoring instances
def get_monitors():
    """Get monitoring instances"""
    global m4max_monitor, container_monitor, trading_monitor, production_dashboard
    
    if not all([m4max_monitor, container_monitor, trading_monitor, production_dashboard]):
        raise HTTPException(status_code=503, detail="Monitoring services not initialized")
    
    return m4max_monitor, container_monitor, trading_monitor, production_dashboard

# Initialize monitoring services
async def initialize_monitoring_services(redis_host: str = "redis", redis_port: int = 6379):
    """Initialize monitoring services"""
    global m4max_monitor, container_monitor, trading_monitor, production_dashboard
    
    try:
        logger.info("Initializing M4 Max monitoring services...")
        
        m4max_monitor = M4MaxHardwareMonitor(redis_host, redis_port)
        container_monitor = ContainerPerformanceMonitor(redis_host, redis_port)
        trading_monitor = TradingPerformanceMonitor(redis_host, redis_port)
        production_dashboard = ProductionMonitoringDashboard(redis_host, redis_port)
        
        # Start background monitoring
        asyncio.create_task(m4max_monitor.start_monitoring(interval=5.0))
        asyncio.create_task(container_monitor.start_monitoring(interval=30.0))
        asyncio.create_task(trading_monitor.start_monitoring(interval=60.0))
        asyncio.create_task(production_dashboard.start_monitoring(interval=15.0))
        
        logger.info("M4 Max monitoring services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring services: {e}")
        raise

# Routes

@router.get("/health", response_model=Dict[str, str])
async def monitoring_health():
    """Get monitoring service health status"""
    try:
        monitors = get_monitors()
        return {
            "status": "healthy",
            "message": "All monitoring services operational",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "m4max_hardware": "running",
                "container_performance": "running", 
                "trading_performance": "running",
                "production_dashboard": "running"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Monitoring services unavailable: {str(e)}")

@router.get("/m4max/hardware/metrics", response_model=M4MaxMetricsResponse)
async def get_m4max_hardware_metrics():
    """Get current M4 Max hardware metrics"""
    try:
        m4max_monitor, _, _, _ = get_monitors()
        
        metrics = m4max_monitor.collect_metrics()
        if not metrics:
            raise HTTPException(status_code=503, detail="Failed to collect M4 Max metrics")
        
        return M4MaxMetricsResponse(
            timestamp=metrics.timestamp,
            cpu_p_cores_usage=metrics.cpu_p_cores_usage,
            cpu_e_cores_usage=metrics.cpu_e_cores_usage,
            cpu_frequency_mhz=metrics.cpu_frequency_mhz,
            unified_memory_usage_gb=metrics.unified_memory_usage_gb,
            unified_memory_bandwidth_gbps=metrics.unified_memory_bandwidth_gbps,
            gpu_utilization_percent=metrics.gpu_utilization_percent,
            neural_engine_utilization_percent=metrics.neural_engine_utilization_percent,
            neural_engine_tops_used=metrics.neural_engine_tops_used,
            thermal_state=metrics.thermal_state,
            power_consumption_watts=metrics.power_consumption_watts
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting M4 Max metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/m4max/hardware/history")
async def get_m4max_hardware_history(limit: int = Query(100, ge=1, le=1000)):
    """Get M4 Max hardware metrics history"""
    try:
        m4max_monitor, _, _, _ = get_monitors()
        
        history = m4max_monitor.get_metrics_history(limit)
        
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting M4 Max history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/containers/metrics", response_model=ContainerMetricsResponse)
async def get_container_metrics():
    """Get current container performance metrics"""
    try:
        _, container_monitor, _, _ = get_monitors()
        
        container_metrics, engine_health = await container_monitor.collect_all_metrics()
        
        summary = container_monitor.get_container_status_summary()
        
        return ContainerMetricsResponse(
            timestamp=datetime.now(),
            containers=[
                {
                    "name": m.container_name,
                    "cpu_usage_percent": m.cpu_usage_percent,
                    "memory_usage_percent": m.memory_usage_percent,
                    "memory_usage_mb": m.memory_usage_mb,
                    "network_rx_mb": m.network_rx_mb,
                    "network_tx_mb": m.network_tx_mb,
                    "uptime_seconds": m.uptime_seconds,
                    "status": m.status.value,
                    "health_status": m.health_status
                } for m in container_metrics
            ],
            engine_health=[
                {
                    "name": h.engine_name,
                    "is_healthy": h.is_healthy,
                    "response_time_ms": h.response_time_ms,
                    "consecutive_failures": h.consecutive_failures,
                    "error_message": h.error_message
                } for h in engine_health
            ],
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error getting container metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/metrics", response_model=TradingMetricsResponse)
async def get_trading_metrics():
    """Get current trading performance metrics"""
    try:
        _, _, trading_monitor, _ = get_monitors()
        
        metrics = await trading_monitor.collect_all_trading_metrics()
        
        return TradingMetricsResponse(
            timestamp=datetime.now(),
            overall_score=metrics.get('overall_score', 0.0),
            latency_metrics=metrics.get('latency_metrics', []),
            throughput_metrics=metrics.get('throughput_metrics', []),
            risk_metrics=metrics.get('risk_metrics', {}),
            ml_metrics=metrics.get('ml_metrics', [])
        )
        
    except Exception as e:
        logger.error(f"Error getting trading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get comprehensive production dashboard summary"""
    try:
        _, _, _, dashboard = get_monitors()
        
        summary = await dashboard.collect_all_metrics()
        
        return {
            "dashboard_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get overall system health status"""
    try:
        _, _, _, dashboard = get_monitors()
        
        summary = dashboard.get_dashboard_summary()
        
        if 'system_health' not in summary:
            raise HTTPException(status_code=503, detail="System health data unavailable")
        
        health = summary['system_health']
        
        return SystemHealthResponse(
            overall_status=health['overall_status'],
            performance_score=health['performance_score'],
            active_alerts=health['active_alerts'],
            critical_alerts=health['critical_alerts'],
            warning_alerts=health['warning_alerts'],
            uptime_seconds=health['uptime_seconds'],
            optimization_opportunities=health.get('optimization_opportunities', []),
            recommendations=health.get('recommendations', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts():
    """Get active alerts"""
    try:
        _, _, _, dashboard = get_monitors()
        
        summary = dashboard.get_dashboard_summary()
        active_alerts = summary.get('active_alerts', [])
        
        return [
            AlertResponse(
                rule_name=alert['rule_name'],
                severity=alert['severity'],
                message=alert['message'],
                timestamp=datetime.fromisoformat(alert['timestamp']),
                resolved=alert.get('resolved', False)
            ) for alert in active_alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledgment: str):
    """Acknowledge an alert"""
    try:
        _, _, _, dashboard = get_monitors()
        
        # Find and acknowledge the alert
        if alert_id in dashboard.active_alerts:
            dashboard.active_alerts[alert_id].acknowledgment = acknowledgment
            logger.info(f"Alert {alert_id} acknowledged: {acknowledgment}")
            return {"status": "acknowledged", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notifications/configure")
async def configure_notifications(config: NotificationConfigRequest):
    """Configure alert notification channels"""
    try:
        _, _, _, dashboard = get_monitors()
        
        # Build email settings if provided
        email_settings = None
        if config.email_smtp_server:
            email_settings = {
                'smtp_server': config.email_smtp_server,
                'smtp_port': config.email_smtp_port or 587,
                'username': config.email_username,
                'password': config.email_password,
                'from_email': config.email_from,
                'to_emails': config.email_to or []
            }
        
        dashboard.configure_notifications(
            slack_webhook=config.slack_webhook,
            email_settings=email_settings,
            pagerduty_key=config.pagerduty_key
        )
        
        return {"status": "configured", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error configuring notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format"""
    try:
        _, _, _, dashboard = get_monitors()
        
        metrics = dashboard.get_prometheus_metrics()
        
        return PlainTextResponse(content=metrics, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/optimizations")
async def get_performance_optimizations():
    """Get performance optimization recommendations"""
    try:
        _, _, _, dashboard = get_monitors()
        
        summary = dashboard.get_dashboard_summary()
        
        if 'system_health' in summary:
            return {
                "optimization_opportunities": summary['system_health'].get('optimization_opportunities', []),
                "recommendations": summary['system_health'].get('recommendations', []),
                "performance_score": summary['system_health'].get('performance_score', 0.0),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "optimization_opportunities": [],
                "recommendations": ["System health data unavailable"],
                "performance_score": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting performance optimizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test/alert")
async def trigger_test_alert(severity: str = "warning", message: str = "Test alert"):
    """Trigger a test alert for notification testing"""
    try:
        _, _, _, dashboard = get_monitors()
        
        from .production_monitoring_dashboard import Alert
        
        test_alert = Alert(
            rule_name="TestAlert",
            severity=AlertSeverity(severity.lower()),
            message=f"Test alert: {message}",
            timestamp=datetime.now()
        )
        
        await dashboard.send_alert_notification(test_alert)
        
        return {
            "status": "test_alert_sent",
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending test alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_monitoring_status():
    """Get detailed monitoring system status"""
    try:
        m4max_monitor, container_monitor, trading_monitor, dashboard = get_monitors()
        
        return {
            "monitoring_services": {
                "m4max_hardware": {
                    "status": "running",
                    "is_m4_max_detected": m4max_monitor.is_m4_max,
                    "monitoring": m4max_monitor.monitoring
                },
                "container_performance": {
                    "status": "running",
                    "monitoring": container_monitor.monitoring
                },
                "trading_performance": {
                    "status": "running", 
                    "monitoring": trading_monitor.monitoring
                },
                "production_dashboard": {
                    "status": "running",
                    "monitoring": dashboard.monitoring,
                    "active_alerts": len(dashboard.active_alerts),
                    "alert_rules": len(dashboard.alert_rules)
                }
            },
            "system_info": {
                "start_time": dashboard.system_start_time.isoformat(),
                "uptime_seconds": (datetime.now() - dashboard.system_start_time).total_seconds()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event handler for FastAPI
async def startup_monitoring():
    """Startup event handler to initialize monitoring"""
    try:
        await initialize_monitoring_services()
        logger.info("Monitoring services started successfully")
    except Exception as e:
        logger.error(f"Failed to start monitoring services: {e}")

# Shutdown event handler for FastAPI  
async def shutdown_monitoring():
    """Shutdown event handler to stop monitoring"""
    try:
        global m4max_monitor, container_monitor, trading_monitor, production_dashboard
        
        if production_dashboard:
            production_dashboard.stop_monitoring()
        if m4max_monitor:
            m4max_monitor.stop_monitoring()
        if container_monitor:
            container_monitor.stop_monitoring()
        if trading_monitor:
            trading_monitor.stop_monitoring()
            
        logger.info("Monitoring services stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping monitoring services: {e}")
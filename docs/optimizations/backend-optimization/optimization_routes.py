"""
FastAPI Routes for CPU Optimization System
==========================================

REST API endpoints for monitoring and controlling the CPU optimization system.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from .optimizer_controller import OptimizerController, OptimizationMode, SystemHealth
from .process_manager import ProcessClass
from .cpu_affinity import WorkloadPriority
from .gcd_scheduler import QoSClass

logger = logging.getLogger(__name__)

# Global optimizer instance
_optimizer_controller: Optional[OptimizerController] = None

def get_optimizer() -> OptimizerController:
    """Get the global optimizer controller instance"""
    global _optimizer_controller
    if _optimizer_controller is None:
        _optimizer_controller = OptimizerController()
        if not _optimizer_controller.initialize():
            raise HTTPException(status_code=500, detail="Failed to initialize optimizer")
    return _optimizer_controller

# Pydantic models for API
class ProcessRegistration(BaseModel):
    pid: int = Field(..., description="Process ID")
    process_class: str = Field(..., description="Process class")
    priority: Optional[str] = Field(None, description="Workload priority")
    preferred_cores: Optional[List[int]] = Field(None, description="Preferred CPU cores")

class WorkloadClassificationRequest(BaseModel):
    function_name: str = Field(..., description="Function name")
    module_name: str = Field(..., description="Module name")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context")

class TaskDispatchRequest(BaseModel):
    queue_name: str = Field(..., description="Dispatch queue name")
    task_name: str = Field(..., description="Task name/identifier")
    qos_class: Optional[str] = Field(None, description="Quality of Service class")

class LatencyMeasurementRequest(BaseModel):
    operation_type: str = Field(..., description="Type of operation being measured")

class OptimizationModeRequest(BaseModel):
    mode: str = Field(..., description="Optimization mode")

class SystemHealthResponse(BaseModel):
    cpu_utilization: float
    memory_utilization: float  
    thermal_state: str
    active_alerts: int
    critical_alerts: int
    optimization_score: float

class ProcessRegistrationResponse(BaseModel):
    success: bool
    message: str
    process_id: int

class WorkloadClassificationResponse(BaseModel):
    category: str
    priority: str
    confidence: float

class LatencyMeasurementResponse(BaseModel):
    operation_id: str
    operation_type: str
    latency_ms: Optional[float] = None

# Create router
router = APIRouter(prefix="/api/v1/optimization", tags=["CPU Optimization"])

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get current system health metrics"""
    try:
        optimizer = get_optimizer()
        health = optimizer.get_system_health()
        
        return SystemHealthResponse(
            cpu_utilization=health.cpu_utilization,
            memory_utilization=health.memory_utilization,
            thermal_state=health.thermal_state,
            active_alerts=health.active_alerts,
            critical_alerts=health.critical_alerts,
            optimization_score=health.optimization_score
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_comprehensive_stats():
    """Get comprehensive system statistics"""
    try:
        optimizer = get_optimizer()
        return optimizer.get_comprehensive_stats()
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/core-utilization")
async def get_core_utilization():
    """Get per-core CPU utilization"""
    try:
        optimizer = get_optimizer()
        return optimizer.cpu_affinity_manager.get_core_utilization()
        
    except Exception as e:
        logger.error(f"Error getting core utilization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register-process", response_model=ProcessRegistrationResponse)
async def register_process(registration: ProcessRegistration):
    """Register a process for optimization"""
    try:
        optimizer = get_optimizer()
        
        # Map string to enum
        try:
            process_class = ProcessClass(registration.process_class.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid process class: {registration.process_class}")
        
        # Map priority if provided
        priority = None
        if registration.priority:
            try:
                priority = WorkloadPriority[registration.priority.upper()]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {registration.priority}")
        
        success = optimizer.register_process(
            registration.pid,
            process_class,
            priority,
            registration.preferred_cores
        )
        
        if success:
            return ProcessRegistrationResponse(
                success=True,
                message="Process registered successfully",
                process_id=registration.pid
            )
        else:
            return ProcessRegistrationResponse(
                success=False,
                message="Failed to register process",
                process_id=registration.pid
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-workload", response_model=WorkloadClassificationResponse)
async def classify_workload(request: WorkloadClassificationRequest):
    """Classify a workload and get optimization recommendations"""
    try:
        optimizer = get_optimizer()
        
        category, priority = optimizer.classify_and_optimize_workload(
            request.function_name,
            request.module_name,
            request.execution_context
        )
        
        return WorkloadClassificationResponse(
            category=category.value,
            priority=priority.name,
            confidence=0.9  # Would need to return this from classify_and_optimize_workload
        )
        
    except Exception as e:
        logger.error(f"Error classifying workload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-latency-measurement", response_model=LatencyMeasurementResponse)
async def start_latency_measurement(request: LatencyMeasurementRequest):
    """Start measuring latency for an operation"""
    try:
        optimizer = get_optimizer()
        operation_id = optimizer.start_latency_measurement(request.operation_type)
        
        return LatencyMeasurementResponse(
            operation_id=operation_id,
            operation_type=request.operation_type
        )
        
    except Exception as e:
        logger.error(f"Error starting latency measurement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/end-latency-measurement/{operation_id}", response_model=LatencyMeasurementResponse)
async def end_latency_measurement(
    operation_id: str,
    success: bool = Query(True, description="Whether the operation was successful"),
    error_msg: Optional[str] = Query(None, description="Error message if operation failed")
):
    """End latency measurement and get results"""
    try:
        optimizer = get_optimizer()
        latency_ms = optimizer.end_latency_measurement(operation_id, success, error_msg)
        
        # Extract operation type from operation_id (simplified)
        operation_type = operation_id.split('_')[0] if '_' in operation_id else "unknown"
        
        return LatencyMeasurementResponse(
            operation_id=operation_id,
            operation_type=operation_type,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Error ending latency measurement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latency-stats")
async def get_latency_stats(
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    duration_minutes: int = Query(5, description="Time window in minutes")
):
    """Get latency statistics"""
    try:
        optimizer = get_optimizer()
        duration_seconds = duration_minutes * 60
        
        return optimizer.performance_monitor.get_latency_stats(
            operation_type, duration_seconds
        )
        
    except Exception as e:
        logger.error(f"Error getting latency stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-mode")
async def get_optimization_mode():
    """Get current optimization mode"""
    try:
        optimizer = get_optimizer()
        return {"mode": optimizer.optimization_mode.value}
        
    except Exception as e:
        logger.error(f"Error getting optimization mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimization-mode")
async def set_optimization_mode(request: OptimizationModeRequest):
    """Set optimization mode"""
    try:
        optimizer = get_optimizer()
        
        # Map string to enum
        try:
            mode = OptimizationMode(request.mode.lower())
        except ValueError:
            valid_modes = [m.value for m in OptimizationMode]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid mode: {request.mode}. Valid modes: {valid_modes}"
            )
        
        success = optimizer.set_optimization_mode(mode)
        
        if success:
            return {"success": True, "mode": mode.value}
        else:
            return {"success": False, "error": "Failed to set optimization mode"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting optimization mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/process-stats")
async def get_process_stats():
    """Get process management statistics"""
    try:
        optimizer = get_optimizer()
        return optimizer.process_manager.get_process_stats()
        
    except Exception as e:
        logger.error(f"Error getting process stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gcd-stats")
async def get_gcd_stats():
    """Get Grand Central Dispatch statistics"""
    try:
        optimizer = get_optimizer()
        return optimizer.gcd_scheduler.get_system_stats()
        
    except Exception as e:
        logger.error(f"Error getting GCD stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gcd-queue-stats/{queue_name}")
async def get_gcd_queue_stats(queue_name: str):
    """Get statistics for a specific GCD queue"""
    try:
        optimizer = get_optimizer()
        return optimizer.gcd_scheduler.get_queue_stats(queue_name)
        
    except Exception as e:
        logger.error(f"Error getting GCD queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workload-classification-stats")
async def get_workload_classification_stats():
    """Get workload classification statistics"""
    try:
        optimizer = get_optimizer()
        return optimizer.workload_classifier.get_classification_stats()
        
    except Exception as e:
        logger.error(f"Error getting classification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear-classification-cache")
async def clear_classification_cache():
    """Clear the workload classification cache"""
    try:
        optimizer = get_optimizer()
        optimizer.workload_classifier.clear_cache()
        return {"success": True, "message": "Classification cache cleared"}
        
    except Exception as e:
        logger.error(f"Error clearing classification cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebalance-workloads")
async def rebalance_workloads():
    """Manually trigger workload rebalancing"""
    try:
        optimizer = get_optimizer()
        stats = optimizer.cpu_affinity_manager.rebalance_workloads()
        return {
            "success": True,
            "processes_moved": stats["moved"],
            "cores_optimized": stats["optimized"]
        }
        
    except Exception as e:
        logger.error(f"Error rebalancing workloads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export-performance-data")
async def export_performance_data(
    background_tasks: BackgroundTasks,
    duration_hours: int = Query(24, description="Duration in hours"),
    output_format: str = Query("json", description="Output format (json)")
):
    """Export performance data for analysis"""
    try:
        optimizer = get_optimizer()
        
        # Create export directory
        timestamp = int(time.time())
        export_dir = f"/tmp/nautilus_performance_export_{timestamp}"
        
        def export_task():
            try:
                success = optimizer.export_performance_data(export_dir, duration_hours)
                if success:
                    logger.info(f"Performance data exported to {export_dir}")
                else:
                    logger.error("Failed to export performance data")
            except Exception as e:
                logger.error(f"Error in export task: {e}")
        
        background_tasks.add_task(export_task)
        
        return {
            "success": True,
            "export_directory": export_dir,
            "message": "Export started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_active_alerts():
    """Get active performance alerts"""
    try:
        optimizer = get_optimizer()
        
        # Get alerts from performance monitor
        active_alerts = []
        for alert in optimizer.performance_monitor.alerts:
            if not alert.resolved:
                active_alerts.append({
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp,
                    "level": alert.level.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold
                })
        
        return {
            "count": len(active_alerts),
            "alerts": active_alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-info")
async def get_system_info():
    """Get system information and capabilities"""
    try:
        optimizer = get_optimizer()
        return optimizer.cpu_affinity_manager.get_system_info()
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time metrics (would need additional setup)
@router.websocket("/ws/metrics")
async def websocket_metrics_endpoint(websocket):
    """WebSocket endpoint for real-time performance metrics"""
    # This would require WebSocket support to be properly implemented
    # Placeholder for now
    pass

# Health check for the optimization system
@router.get("/ping")
async def ping():
    """Health check endpoint"""
    try:
        optimizer = get_optimizer()
        if optimizer.is_running:
            return {"status": "healthy", "timestamp": time.time()}
        else:
            return {"status": "unhealthy", "timestamp": time.time()}
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Optimization system unavailable")

# Admin endpoints (would require authentication in production)
@router.post("/admin/shutdown")
async def admin_shutdown(background_tasks: BackgroundTasks):
    """Shutdown the optimization system (admin only)"""
    try:
        global _optimizer_controller
        if _optimizer_controller:
            def shutdown_task():
                _optimizer_controller.shutdown()
                _optimizer_controller = None
            
            background_tasks.add_task(shutdown_task)
            return {"success": True, "message": "Shutdown initiated"}
        else:
            return {"success": False, "message": "No optimizer instance to shutdown"}
            
    except Exception as e:
        logger.error(f"Error during admin shutdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/restart")  
async def admin_restart(background_tasks: BackgroundTasks):
    """Restart the optimization system (admin only)"""
    try:
        global _optimizer_controller
        
        def restart_task():
            global _optimizer_controller
            if _optimizer_controller:
                _optimizer_controller.shutdown()
            
            _optimizer_controller = OptimizerController()
            if not _optimizer_controller.initialize():
                logger.error("Failed to restart optimizer")
                _optimizer_controller = None
        
        background_tasks.add_task(restart_task)
        return {"success": True, "message": "Restart initiated"}
        
    except Exception as e:
        logger.error(f"Error during admin restart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Container-specific optimization endpoints
@router.get("/containers/stats")
async def get_container_stats():
    """Get container CPU optimization statistics"""
    try:
        optimizer = get_optimizer()
        if not optimizer.container_optimizer:
            raise HTTPException(status_code=503, detail="Container optimizer not available")
        
        return optimizer.container_optimizer.get_container_stats()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting container stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/containers/performance-analysis")
async def get_container_performance_analysis():
    """Get detailed container performance analysis"""
    try:
        optimizer = get_optimizer()
        if not optimizer.container_optimizer:
            raise HTTPException(status_code=503, detail="Container optimizer not available")
        
        return optimizer.container_optimizer.get_container_performance_analysis()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting container performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/containers/optimize")
async def force_container_optimization(
    container_name: Optional[str] = Query(None, description="Specific container to optimize")
):
    """Force container CPU optimization"""
    try:
        optimizer = get_optimizer()
        if not optimizer.container_optimizer:
            raise HTTPException(status_code=503, detail="Container optimizer not available")
        
        result = optimizer.container_optimizer.force_container_optimization(container_name)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing container optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ContainerPriorityUpdate(BaseModel):
    priority: str = Field(..., description="New container priority")

@router.post("/containers/{container_name}/priority")
async def update_container_priority(
    container_name: str,
    request: ContainerPriorityUpdate
):
    """Update container priority"""
    try:
        optimizer = get_optimizer()
        if not optimizer.container_optimizer:
            raise HTTPException(status_code=503, detail="Container optimizer not available")
        
        # Map string to enum
        try:
            from .container_cpu_optimizer import ContainerPriority
            new_priority = ContainerPriority[request.priority.upper()]
        except KeyError:
            valid_priorities = [p.name for p in ContainerPriority]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid priority: {request.priority}. Valid priorities: {valid_priorities}"
            )
        
        success = optimizer.container_optimizer.update_container_priority(
            container_name, new_priority
        )
        
        if success:
            return {
                "success": True,
                "container_name": container_name,
                "new_priority": new_priority.name,
                "message": "Container priority updated successfully"
            }
        else:
            return {
                "success": False,
                "container_name": container_name,
                "error": "Failed to update container priority"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating container priority: {e}")
        raise HTTPException(status_code=500, detail=str(e))
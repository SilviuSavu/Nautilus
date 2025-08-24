"""
Phase 8 Security Monitoring Dashboard
Real-time monitoring and visualization for autonomous security operations.

Provides FastAPI endpoints for monitoring Phase 8 security services with
real-time status updates, metrics visualization, and alert management.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import logging

from phase8_startup_service import (
    phase8_startup_service,
    get_phase8_health,
    get_phase8_metrics,
    ServiceStatus,
    HealthStatus,
    ServiceMetrics
)
from messagebus_client import messagebus_client, MessageBusMessage

logger = logging.getLogger(__name__)


class Phase8MonitoringDashboard:
    """Real-time monitoring dashboard for Phase 8 security operations"""
    
    def __init__(self):
        self.active_websockets: List[WebSocket] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_size = 1000
        
    async def add_websocket(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.active_websockets.append(websocket)
        logger.info(f"New dashboard connection. Total connections: {len(self.active_websockets)}")
        
        # Send initial state
        await self.send_initial_state(websocket)
    
    async def remove_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)
        logger.info(f"Dashboard connection closed. Total connections: {len(self.active_websockets)}")
    
    async def send_initial_state(self, websocket: WebSocket):
        """Send initial dashboard state to new connection"""
        try:
            initial_data = {
                "type": "initial_state",
                "timestamp": datetime.utcnow().isoformat(),
                "system_health": get_phase8_health(),
                "metrics": get_phase8_metrics(),
                "recent_alerts": self.alert_history[-50:] if self.alert_history else []
            }
            await websocket.send_text(json.dumps(initial_data))
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.active_websockets:
            return
        
        message = {
            "type": update_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Remove disconnected websockets
        active_connections = []
        for websocket in self.active_websockets:
            try:
                await websocket.send_text(json.dumps(message))
                active_connections.append(websocket)
            except Exception as e:
                logger.warning(f"Failed to send to websocket: {e}")
        
        self.active_websockets = active_connections
    
    async def add_alert(self, alert_data: Dict[str, Any]):
        """Add security alert to history and broadcast"""
        alert_with_timestamp = {
            **alert_data,
            "timestamp": datetime.utcnow().isoformat(),
            "id": len(self.alert_history)
        }
        
        self.alert_history.append(alert_with_timestamp)
        
        # Keep history size manageable
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        await self.broadcast_update("new_alert", alert_with_timestamp)
    
    async def update_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Update service metrics and broadcast"""
        if service_name not in self.metrics_history:
            self.metrics_history[service_name] = []
        
        metric_point = {
            "timestamp": datetime.utcnow().isoformat(),
            **metrics
        }
        
        self.metrics_history[service_name].append(metric_point)
        
        # Keep history size manageable
        if len(self.metrics_history[service_name]) > self.max_history_size:
            self.metrics_history[service_name] = self.metrics_history[service_name][-self.max_history_size:]
        
        await self.broadcast_update("metrics_update", {
            "service": service_name,
            "metrics": metric_point
        })
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        category_counts = {}
        
        for alert in recent_alerts:
            severity = alert.get("severity", "unknown").lower()
            category = alert.get("category", "unknown")
            
            if severity in severity_counts:
                severity_counts[severity] += 1
            
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_alerts": len(recent_alerts),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "time_period_hours": hours,
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }
    
    def get_service_metrics_summary(self, service_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get service metrics summary for specified time period"""
        if service_name not in self.metrics_history:
            return {"error": f"No metrics found for service {service_name}"}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.metrics_history[service_name]
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": f"No recent metrics for service {service_name}"}
        
        # Calculate averages
        numeric_fields = ["cpu_usage", "memory_usage_mb", "threat_alerts_processed", "response_actions_executed"]
        averages = {}
        
        for field in numeric_fields:
            values = [m.get(field, 0) for m in recent_metrics if isinstance(m.get(field), (int, float))]
            if values:
                averages[f"avg_{field}"] = sum(values) / len(values)
                averages[f"max_{field}"] = max(values)
                averages[f"min_{field}"] = min(values)
        
        return {
            "service_name": service_name,
            "time_period_hours": hours,
            "data_points": len(recent_metrics),
            "averages": averages,
            "latest_metrics": recent_metrics[-1] if recent_metrics else None
        }


# Global dashboard instance
dashboard = Phase8MonitoringDashboard()


# FastAPI routes for monitoring dashboard
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await dashboard.add_websocket(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Handle client requests
            try:
                request = json.loads(data)
                if request.get("type") == "refresh":
                    await dashboard.send_initial_state(websocket)
                elif request.get("type") == "get_alerts":
                    hours = request.get("hours", 24)
                    alert_summary = dashboard.get_alert_summary(hours)
                    await websocket.send_text(json.dumps({
                        "type": "alert_summary",
                        "data": alert_summary
                    }))
            except json.JSONDecodeError:
                # Ignore invalid JSON
                pass
            
    except WebSocketDisconnect:
        await dashboard.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        await dashboard.remove_websocket(websocket)


async def get_dashboard_status():
    """Get current dashboard status"""
    return {
        "active_connections": len(dashboard.active_websockets),
        "total_alerts": len(dashboard.alert_history),
        "tracked_services": list(dashboard.metrics_history.keys()),
        "system_health": get_phase8_health(),
        "last_updated": datetime.utcnow().isoformat()
    }


async def get_alert_summary_endpoint(hours: int = Query(24, ge=1, le=168)):
    """Get alert summary for specified hours"""
    return dashboard.get_alert_summary(hours)


async def get_service_metrics_endpoint(service_name: str, hours: int = Query(1, ge=1, le=24)):
    """Get service metrics summary"""
    return dashboard.get_service_metrics_summary(service_name, hours)


async def get_all_alerts_endpoint(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """Get all alerts with optional filtering"""
    alerts = dashboard.alert_history.copy()
    
    # Apply filters
    if severity:
        alerts = [a for a in alerts if a.get("severity", "").lower() == severity.lower()]
    
    if category:
        alerts = [a for a in alerts if a.get("category", "").lower() == category.lower()]
    
    # Sort by timestamp (newest first) and limit
    alerts.sort(key=lambda x: x["timestamp"], reverse=True)
    alerts = alerts[:limit]
    
    return {
        "alerts": alerts,
        "total_count": len(dashboard.alert_history),
        "filtered_count": len(alerts),
        "filters": {
            "severity": severity,
            "category": category,
            "limit": limit
        }
    }


async def trigger_test_alert_endpoint():
    """Trigger a test alert for dashboard testing"""
    test_alert = {
        "severity": "medium",
        "category": "test_alert",
        "title": "Dashboard Test Alert",
        "description": "This is a test alert for dashboard functionality",
        "source": "dashboard_test",
        "details": {
            "test_parameter": "test_value",
            "alert_id": f"test_{datetime.utcnow().timestamp()}"
        }
    }
    
    await dashboard.add_alert(test_alert)
    return {"message": "Test alert triggered", "alert": test_alert}


# Background task to monitor Phase 8 services
async def monitor_phase8_services():
    """Background task to monitor Phase 8 services and update dashboard"""
    while True:
        try:
            # Get current system health
            health_data = get_phase8_health()
            await dashboard.broadcast_update("health_update", health_data)
            
            # Get detailed metrics for each service
            metrics_data = get_phase8_metrics()
            if "service_metrics" in metrics_data:
                for service_name, service_metrics in metrics_data["service_metrics"].items():
                    await dashboard.update_metrics(service_name, service_metrics.__dict__ if hasattr(service_metrics, '__dict__') else service_metrics)
            
            # Check for new alerts from Phase 8 services
            # This would be implemented based on how Phase 8 services expose alerts
            # For now, we'll simulate checking service status for degradation
            for service_name, metrics in metrics_data.get("service_metrics", {}).items():
                if hasattr(metrics, 'health') and metrics.health == HealthStatus.UNHEALTHY:
                    await dashboard.add_alert({
                        "severity": "high",
                        "category": "service_health",
                        "title": f"Service {service_name} Unhealthy",
                        "description": f"Service {service_name} is reporting unhealthy status",
                        "source": "health_monitor",
                        "service": service_name,
                        "details": metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
                    })
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in Phase 8 monitoring loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# HTML dashboard template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 8 Security Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #0a0a0a;
            color: #00ff00;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 20px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 255, 0, 0.1);
        }
        .status-card h3 {
            margin-top: 0;
            color: #00ff00;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        .health-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .healthy { background-color: #00ff00; }
        .degraded { background-color: #ffaa00; }
        .unhealthy { background-color: #ff0000; }
        .alerts-section {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .alert {
            background-color: #2a2a2a;
            border-left: 4px solid;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .alert.low { border-left-color: #00aa00; }
        .alert.medium { border-left-color: #ffaa00; }
        .alert.high { border-left-color: #ff6600; }
        .alert.critical { border-left-color: #ff0000; }
        .timestamp {
            color: #888;
            font-size: 0.8em;
        }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
        }
        .connected { background-color: #00aa00; color: white; }
        .disconnected { background-color: #aa0000; color: white; }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="dashboard-container">
        <div class="header">
            <h1>Phase 8 Security Operations Dashboard</h1>
            <p>Real-time monitoring of autonomous security services</p>
        </div>
        
        <div class="status-grid" id="statusGrid">
            <!-- Service status cards will be populated here -->
        </div>
        
        <div class="alerts-section">
            <h3>Recent Security Alerts</h3>
            <div id="alertsContainer">
                <!-- Alerts will be populated here -->
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/phase8/dashboard/ws`);
            
            ws.onopen = function() {
                console.log('Dashboard WebSocket connected');
                document.getElementById('connectionStatus').textContent = 'Connected';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                reconnectAttempts = 0;
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
            
            ws.onclose = function() {
                console.log('Dashboard WebSocket disconnected');
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    setTimeout(() => {
                        reconnectAttempts++;
                        connect();
                    }, 5000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('Dashboard WebSocket error:', error);
            };
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'initial_state':
                    updateSystemHealth(data.system_health);
                    updateAlerts(data.recent_alerts || []);
                    break;
                case 'health_update':
                    updateSystemHealth(data.data);
                    break;
                case 'new_alert':
                    addAlert(data.data);
                    break;
                case 'metrics_update':
                    updateServiceMetrics(data.data);
                    break;
            }
        }

        function updateSystemHealth(healthData) {
            const statusGrid = document.getElementById('statusGrid');
            statusGrid.innerHTML = '';
            
            // Overall system health
            const overallCard = document.createElement('div');
            overallCard.className = 'status-card';
            overallCard.innerHTML = `
                <h3>System Health</h3>
                <p>
                    <span class="health-indicator ${healthData.overall_health}"></span>
                    ${healthData.overall_health.toUpperCase()}
                </p>
                <p>Services: ${healthData.healthy_services}/${healthData.total_services} healthy</p>
                <p class="timestamp">Last check: ${new Date().toLocaleTimeString()}</p>
            `;
            statusGrid.appendChild(overallCard);
            
            // Individual service status
            if (healthData.services) {
                for (const [serviceName, serviceData] of Object.entries(healthData.services)) {
                    const serviceCard = document.createElement('div');
                    serviceCard.className = 'status-card';
                    serviceCard.innerHTML = `
                        <h3>${serviceName.replace('_', ' ').toUpperCase()}</h3>
                        <p>
                            <span class="health-indicator ${serviceData.health}"></span>
                            ${serviceData.health.toUpperCase()}
                        </p>
                        <p>Status: ${serviceData.status}</p>
                        <p>Uptime: ${Math.floor(serviceData.uptime_seconds / 60)} minutes</p>
                        ${serviceData.last_error ? `<p style="color: #ff6666;">Last Error: ${serviceData.last_error}</p>` : ''}
                    `;
                    statusGrid.appendChild(serviceCard);
                }
            }
        }

        function updateAlerts(alerts) {
            const alertsContainer = document.getElementById('alertsContainer');
            alertsContainer.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No recent alerts</p>';
                return;
            }
            
            alerts.forEach(alert => addAlert(alert, false));
        }

        function addAlert(alert, prepend = true) {
            const alertsContainer = document.getElementById('alertsContainer');
            const alertElement = document.createElement('div');
            alertElement.className = `alert ${alert.severity}`;
            alertElement.innerHTML = `
                <strong>${alert.title || 'Security Alert'}</strong>
                <p>${alert.description}</p>
                <p class="timestamp">${new Date(alert.timestamp).toLocaleString()}</p>
                <p><em>Category: ${alert.category} | Source: ${alert.source}</em></p>
            `;
            
            if (prepend && alertsContainer.firstChild) {
                alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
            } else {
                alertsContainer.appendChild(alertElement);
            }
            
            // Keep only last 20 alerts visible
            while (alertsContainer.children.length > 20) {
                alertsContainer.removeChild(alertsContainer.lastChild);
            }
        }

        function updateServiceMetrics(data) {
            // Update metrics in service cards if needed
            console.log('Service metrics update:', data);
        }

        // Initialize connection
        connect();
        
        // Refresh data every 30 seconds
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'refresh'}));
            }
        }, 30000);
    </script>
</body>
</html>
"""

async def dashboard_html_endpoint():
    """Serve the HTML dashboard"""
    return HTMLResponse(content=DASHBOARD_HTML)


# Function to setup dashboard routes in main FastAPI app
def setup_phase8_dashboard_routes(app: FastAPI):
    """Setup Phase 8 dashboard routes in the main FastAPI application"""
    
    # HTML Dashboard
    app.add_api_route(
        "/phase8/dashboard", 
        dashboard_html_endpoint, 
        methods=["GET"], 
        response_class=HTMLResponse,
        tags=["Phase 8 Dashboard"]
    )
    
    # WebSocket for real-time updates
    app.add_websocket_route("/api/v1/phase8/dashboard/ws", dashboard_websocket_endpoint)
    
    # API endpoints
    app.add_api_route(
        "/api/v1/phase8/dashboard/status", 
        get_dashboard_status, 
        methods=["GET"],
        tags=["Phase 8 Dashboard"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/dashboard/alerts/summary", 
        get_alert_summary_endpoint, 
        methods=["GET"],
        tags=["Phase 8 Dashboard"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/dashboard/service/{service_name}/metrics", 
        get_service_metrics_endpoint, 
        methods=["GET"],
        tags=["Phase 8 Dashboard"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/dashboard/alerts", 
        get_all_alerts_endpoint, 
        methods=["GET"],
        tags=["Phase 8 Dashboard"]
    )
    
    app.add_api_route(
        "/api/v1/phase8/dashboard/test-alert", 
        trigger_test_alert_endpoint, 
        methods=["POST"],
        tags=["Phase 8 Dashboard"]
    )
    
    # Start background monitoring
    asyncio.create_task(monitor_phase8_services())
    
    logger.info("Phase 8 Dashboard routes configured successfully")


if __name__ == "__main__":
    """Standalone dashboard server for development"""
    import uvicorn
    
    app = FastAPI(title="Phase 8 Security Dashboard", version="1.0.0")
    setup_phase8_dashboard_routes(app)
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
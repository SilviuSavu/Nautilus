#!/usr/bin/env python3
"""
Engine Network API
Extended API endpoints for engine network visualization and management.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import uuid

from engines.common.nautilus_environment import get_nautilus_environment
from engines.common.engine_identity import ProcessingCapability, EngineRole
from engines.common.partnership_manager import PartnershipType, PartnershipStatus


logger = logging.getLogger(__name__)


# Pydantic models for API
class EngineStatusModel(BaseModel):
    """Engine status model"""
    engine_id: str
    name: str
    port: int
    status: str
    roles: List[str]
    capabilities: List[str]
    performance: Dict[str, Any]
    health: Dict[str, Any]
    partnerships: List[Dict[str, Any]] = Field(default_factory=list)


class NetworkNodeModel(BaseModel):
    """Network node model"""
    id: str
    name: str
    type: str  # "engine", "coordinator", "messagebus"
    status: str
    properties: Dict[str, Any]
    position: Optional[Dict[str, float]] = None


class NetworkEdgeModel(BaseModel):
    """Network edge model"""
    id: str
    source: str
    target: str
    type: str  # "partnership", "message_flow", "dependency"
    strength: float
    properties: Dict[str, Any]


class NetworkTopologyModel(BaseModel):
    """Complete network topology model"""
    nodes: List[NetworkNodeModel]
    edges: List[NetworkEdgeModel]
    clusters: Dict[str, List[str]]
    metadata: Dict[str, Any]


class WorkflowExecutionModel(BaseModel):
    """Workflow execution model"""
    workflow_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PartnershipProposalModel(BaseModel):
    """Partnership proposal model"""
    target_engine_id: str
    partnership_type: str
    expected_latency_ms: float = 10.0
    expected_throughput: int = 1000
    reliability_requirement: float = 0.95
    description: str = ""


class EngineNetworkAPI:
    """Extended API for engine network management"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.websocket_connections: List[WebSocket] = []
        self._notification_task: Optional[asyncio.Task] = None
        
        # Real-time data
        self._live_metrics: Dict[str, Any] = {}
        self._live_events: List[Dict[str, Any]] = []
        self._max_events = 1000
    
    async def start_real_time_updates(self):
        """Start real-time update broadcasting"""
        if self._notification_task is None:
            self._notification_task = asyncio.create_task(self._broadcast_updates())
    
    async def stop_real_time_updates(self):
        """Stop real-time update broadcasting"""
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
            self._notification_task = None
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket connection added. Total: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.info(f"WebSocket connection removed. Total: {len(self.websocket_connections)}")
    
    async def _broadcast_updates(self):
        """Broadcast real-time updates to WebSocket connections"""
        while True:
            try:
                if self.websocket_connections:
                    # Collect current metrics
                    metrics = await self._collect_live_metrics()
                    
                    # Broadcast to all connections
                    disconnected = []
                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_json({
                                "type": "metrics_update",
                                "data": metrics,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception:
                            disconnected.append(websocket)
                    
                    # Remove disconnected clients
                    for ws in disconnected:
                        await self.remove_websocket_connection(ws)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast updates: {e}")
                await asyncio.sleep(10)
    
    async def _collect_live_metrics(self) -> Dict[str, Any]:
        """Collect current live metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": {},
            "engine_metrics": {},
            "partnership_metrics": {},
            "message_flows": [],
            "alerts": []
        }
        
        try:
            # System status
            system_status = self.coordinator.get_system_status()
            metrics["system_status"] = {
                "total_engines": system_status.total_engines,
                "online_engines": system_status.online_engines,
                "active_partnerships": system_status.active_partnerships,
                "active_tasks": system_status.active_tasks,
                "system_health": system_status.system_health
            }
            
            # Engine metrics
            if self.coordinator.discovery_protocol:
                online_engines = self.coordinator.discovery_protocol.get_online_engines()
                for engine_id, engine_data in online_engines.items():
                    performance = engine_data.get("performance", {})
                    health = engine_data.get("health", {})
                    
                    metrics["engine_metrics"][engine_id] = {
                        "response_time_ms": performance.get("response_time_ms", 0),
                        "throughput": performance.get("throughput", 0),
                        "status": health.get("status", "unknown"),
                        "load_pct": health.get("load_pct", 0),
                        "uptime": health.get("uptime_seconds", 0)
                    }
            
            # Partnership metrics
            if self.coordinator.partnership_manager:
                partnership_stats = self.coordinator.partnership_manager.get_partnership_statistics()
                metrics["partnership_metrics"] = partnership_stats
            
            # Generate alerts for critical issues
            alerts = []
            for engine_id, engine_metrics in metrics["engine_metrics"].items():
                if engine_metrics["load_pct"] > 80:
                    alerts.append({
                        "type": "warning",
                        "source": engine_id,
                        "message": f"High CPU load: {engine_metrics['load_pct']:.1f}%",
                        "timestamp": datetime.now().isoformat()
                    })
                
                if engine_metrics["response_time_ms"] > 1000:
                    alerts.append({
                        "type": "warning",
                        "source": engine_id,
                        "message": f"Slow response time: {engine_metrics['response_time_ms']:.1f}ms",
                        "timestamp": datetime.now().isoformat()
                    })
            
            metrics["alerts"] = alerts
            
        except Exception as e:
            logger.error(f"Error collecting live metrics: {e}")
        
        return metrics
    
    def add_event(self, event_type: str, source: str, message: str, data: Dict[str, Any] = None):
        """Add event to live feed"""
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "source": source,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._live_events.append(event)
        
        # Keep only recent events
        if len(self._live_events) > self._max_events:
            self._live_events = self._live_events[-self._max_events:]
        
        # Broadcast event to WebSocket connections
        asyncio.create_task(self._broadcast_event(event))
    
    async def _broadcast_event(self, event: Dict[str, Any]):
        """Broadcast single event to WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "event",
            "data": event
        }
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            await self.remove_websocket_connection(ws)
    
    # API Methods
    def get_enhanced_network_topology(self) -> NetworkTopologyModel:
        """Get enhanced network topology with additional metadata"""
        basic_topology = self.coordinator.get_network_topology()
        
        nodes = []
        edges = []
        
        # Convert to enhanced models
        for node_data in basic_topology.get("nodes", []):
            node = NetworkNodeModel(
                id=node_data["id"],
                name=node_data["name"],
                type="engine",
                status=node_data["status"],
                properties={
                    "port": node_data["port"],
                    "roles": node_data["roles"],
                    "capabilities": node_data["capabilities"],
                    "performance": node_data["performance"],
                    "health": node_data["health"]
                }
            )
            nodes.append(node)
        
        # Add message bus nodes
        env = get_nautilus_environment()
        for bus in env.message_buses:
            node = NetworkNodeModel(
                id=f"messagebus_{bus.port}",
                name=f"{bus.bus_type.value.replace('_', ' ').title()}",
                type="messagebus",
                status="active",
                properties={
                    "port": bus.port,
                    "optimization": bus.optimization,
                    "purpose": bus.purpose,
                    "container_name": bus.container_name
                }
            )
            nodes.append(node)
        
        for edge_data in basic_topology.get("edges", []):
            edge = NetworkEdgeModel(
                id=f"{edge_data['source']}_{edge_data['target']}",
                source=edge_data["source"],
                target=edge_data["target"],
                type=edge_data["type"],
                strength=edge_data["strength"],
                properties={
                    "status": edge_data["status"],
                    "performance_score": edge_data["performance_score"],
                    "message_count": edge_data["message_count"]
                }
            )
            edges.append(edge)
        
        return NetworkTopologyModel(
            nodes=nodes,
            edges=edges,
            clusters=basic_topology.get("clusters", {}),
            metadata={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "generated_at": datetime.now().isoformat(),
                "engine_types": list(set(node.type for node in nodes))
            }
        )
    
    def get_engine_performance_metrics(self, engine_id: str) -> Dict[str, Any]:
        """Get detailed performance metrics for specific engine"""
        engine_details = self.coordinator.get_engine_details(engine_id)
        if not engine_details:
            raise HTTPException(status_code=404, detail="Engine not found")
        
        # Get routing metrics
        routing_metrics = {}
        if self.coordinator.message_router:
            route_metrics = self.coordinator.message_router.route_metrics.get(engine_id)
            if route_metrics:
                routing_metrics = {
                    "total_messages": route_metrics.total_messages,
                    "success_rate": route_metrics.success_rate,
                    "average_response_time_ms": route_metrics.average_response_time_ms,
                    "overall_score": route_metrics.overall_score
                }
        
        # Get partnership performance
        partnership_performance = {}
        if self.coordinator.partnership_manager:
            partnership = self.coordinator.partnership_manager.get_partnership(engine_id)
            if partnership:
                partnership_performance = {
                    "partnership_type": partnership.partnership_type.value,
                    "status": partnership.status.value,
                    "performance_score": partnership.performance_score,
                    "relationship_strength": partnership.relationship_strength,
                    "message_count": partnership.message_count,
                    "success_rate": partnership.success_rate
                }
        
        return {
            "engine_id": engine_id,
            "basic_metrics": engine_details.get("performance", {}),
            "health_metrics": engine_details.get("health", {}),
            "routing_metrics": routing_metrics,
            "partnership_performance": partnership_performance,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        summary = {
            "overall_health": "healthy",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "statistics": {},
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            # System status
            system_status = self.coordinator.get_system_status()
            summary["statistics"]["system"] = {
                "total_engines": system_status.total_engines,
                "online_engines": system_status.online_engines,
                "online_ratio": system_status.online_engines / max(system_status.total_engines, 1),
                "active_partnerships": system_status.active_partnerships,
                "active_tasks": system_status.active_tasks
            }
            
            # Check for critical issues
            online_ratio = summary["statistics"]["system"]["online_ratio"]
            if online_ratio < 0.7:
                summary["overall_health"] = "critical"
                summary["critical_issues"].append({
                    "type": "low_availability",
                    "message": f"Only {system_status.online_engines}/{system_status.total_engines} engines online",
                    "severity": "critical"
                })
            elif online_ratio < 0.9:
                summary["overall_health"] = "degraded"
                summary["warnings"].append({
                    "type": "reduced_availability",
                    "message": f"{system_status.total_engines - system_status.online_engines} engines offline",
                    "severity": "warning"
                })
            
            # Engine-specific health
            if self.coordinator.discovery_protocol:
                online_engines = self.coordinator.discovery_protocol.get_online_engines()
                engine_issues = []
                
                for engine_id, engine_data in online_engines.items():
                    health = engine_data.get("health", {})
                    performance = engine_data.get("performance", {})
                    
                    load_pct = health.get("load_pct", 0)
                    response_time = performance.get("response_time_ms", 0)
                    
                    if load_pct > 90:
                        engine_issues.append({
                            "engine_id": engine_id,
                            "type": "high_load",
                            "message": f"CPU load: {load_pct:.1f}%",
                            "severity": "critical" if load_pct > 95 else "warning"
                        })
                    
                    if response_time > 1000:
                        engine_issues.append({
                            "engine_id": engine_id,
                            "type": "slow_response",
                            "message": f"Response time: {response_time:.1f}ms",
                            "severity": "warning"
                        })
                
                # Categorize issues
                for issue in engine_issues:
                    if issue["severity"] == "critical":
                        summary["critical_issues"].append(issue)
                        if summary["overall_health"] == "healthy":
                            summary["overall_health"] = "degraded"
                    else:
                        summary["warnings"].append(issue)
            
            # Partnership health
            if self.coordinator.partnership_manager:
                partnership_stats = self.coordinator.partnership_manager.get_partnership_statistics()
                avg_performance = partnership_stats.get("average_performance_score", 0)
                
                if avg_performance < 0.5:
                    summary["warnings"].append({
                        "type": "poor_partnership_performance",
                        "message": f"Average partnership performance: {avg_performance:.2f}",
                        "severity": "warning"
                    })
                
                summary["statistics"]["partnerships"] = {
                    "total": partnership_stats.get("total_partnerships", 0),
                    "active": partnership_stats.get("active_partnerships", 0),
                    "average_performance": avg_performance
                }
            
            # Generate recommendations
            recommendations = []
            
            if len(summary["critical_issues"]) > 0:
                recommendations.append({
                    "type": "investigate_critical_issues",
                    "message": "Investigate and resolve critical system issues",
                    "priority": "high"
                })
            
            if summary["statistics"]["system"]["online_ratio"] < 0.9:
                recommendations.append({
                    "type": "restart_offline_engines",
                    "message": "Consider restarting offline engines",
                    "priority": "medium"
                })
            
            if len(summary["warnings"]) > 5:
                recommendations.append({
                    "type": "system_optimization",
                    "message": "System showing multiple warning signs, consider optimization",
                    "priority": "medium"
                })
            
            summary["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating system health summary: {e}")
            summary["overall_health"] = "unknown"
            summary["critical_issues"].append({
                "type": "monitoring_error",
                "message": f"Error collecting health data: {str(e)}",
                "severity": "critical"
            })
        
        return summary
    
    def get_live_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent live events"""
        return self._live_events[-limit:] if limit > 0 else self._live_events
    
    async def propose_partnership(self, proposal: PartnershipProposalModel) -> Dict[str, Any]:
        """Propose new partnership"""
        if not self.coordinator.partnership_manager:
            raise HTTPException(status_code=503, detail="Partnership manager not available")
        
        # Validate partnership type
        try:
            partnership_type = PartnershipType(proposal.partnership_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid partnership type")
        
        # Check if engine exists
        engine_details = self.coordinator.get_engine_details(proposal.target_engine_id)
        if not engine_details:
            raise HTTPException(status_code=404, detail="Target engine not found")
        
        # Create partnership
        partnership = self.coordinator.partnership_manager.create_partnership(
            proposal.target_engine_id,
            partnership_type,
            proposal.expected_latency_ms,
            proposal.expected_throughput,
            proposal.reliability_requirement
        )
        
        # Add event
        self.add_event(
            "partnership_created",
            self.coordinator.coordinator_identity.engine_id,
            f"Created {partnership_type.value} partnership with {proposal.target_engine_id}",
            {"partnership_id": partnership.partnership_id}
        )
        
        return {
            "partnership_id": partnership.partnership_id,
            "status": "created",
            "details": {
                "target_engine": proposal.target_engine_id,
                "type": partnership_type.value,
                "expected_latency_ms": proposal.expected_latency_ms,
                "expected_throughput": proposal.expected_throughput,
                "reliability_requirement": proposal.reliability_requirement
            },
            "created_at": datetime.now().isoformat()
        }
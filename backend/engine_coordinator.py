#!/usr/bin/env python3
"""
Engine Coordinator Service
Central orchestration service that manages all engine interactions, discovery, and coordination.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from engines.common.nautilus_environment import get_nautilus_environment
from engines.common.engine_identity import (
    EngineIdentity, EngineRole, ProcessingCapability, create_ml_engine_identity, create_risk_engine_identity
)
from engines.common.engine_discovery import EngineDiscoveryProtocol, EngineRegistry
from engines.common.intelligent_router import MessageRouter, WorkflowTemplates, TaskPriority
from engines.common.partnership_manager import PartnershipManager, PartnershipType
from engines.common.ai_agent_specialists import AIAgentCoordinator


logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Overall system status"""
    total_engines: int
    online_engines: int
    active_partnerships: int
    active_tasks: int
    system_health: str  # "healthy", "degraded", "critical"
    last_updated: str
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class EngineCoordinator:
    """Central coordinator for all engine interactions"""
    
    def __init__(self):
        # Core components
        self.environment = get_nautilus_environment()
        self.engine_registry = EngineRegistry()
        
        # Coordinator's own identity (acts as meta-engine)
        self.coordinator_identity = self._create_coordinator_identity()
        
        # Discovery and routing
        self.discovery_protocol: Optional[EngineDiscoveryProtocol] = None
        self.message_router: Optional[MessageRouter] = None
        self.partnership_manager: Optional[PartnershipManager] = None
        
        # AI Agent Specialists
        self.ai_agent_coordinator: Optional[AIAgentCoordinator] = None
        
        # System state
        self.system_metrics = {
            "startup_time": datetime.now().isoformat(),
            "total_messages_routed": 0,
            "total_tasks_completed": 0,
            "average_response_time": 0.0,
            "engine_discoveries": 0,
            "partnerships_established": 0
        }
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._auto_management_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Cached data for API responses
        self._cached_network_topology: Dict[str, Any] = {}
        self._cached_performance_stats: Dict[str, Any] = {}
        self._last_cache_update = datetime.now()
        self._cache_ttl_seconds = 30
    
    def _create_coordinator_identity(self) -> EngineIdentity:
        """Create identity for the coordinator itself"""
        from engines.common.engine_identity import EngineCapabilities, DataSchema, DataFormat, PartnershipPreference
        
        capabilities = EngineCapabilities(
            supported_roles=[EngineRole.ANALYTICS, EngineRole.DATA_PROCESSOR],
            processing_capabilities=[
                ProcessingCapability.REAL_TIME_STREAMING,
                ProcessingCapability.BATCH_PROCESSING
            ],
            input_data_schemas=[
                DataSchema(
                    name="coordination_messages",
                    format=DataFormat.JSON,
                    required_fields=["source", "target", "message_type"],
                    description="Inter-engine coordination messages"
                )
            ],
            output_data_schemas=[
                DataSchema(
                    name="coordination_responses",
                    format=DataFormat.JSON,
                    required_fields=["status", "result"],
                    description="Coordination operation results"
                )
            ],
            partnership_preferences=[],  # Coordinator partners with all engines
            hardware_requirements=["performance_cores"],
            software_dependencies=["fastapi", "redis", "postgresql"]
        )
        
        return EngineIdentity(
            engine_id="ENGINE_COORDINATOR",
            engine_name="Engine Coordinator",
            engine_port=8000,  # Coordinator API port
            capabilities=capabilities,
            version="2025.1.0"
        )
    
    async def initialize(self):
        """Initialize the coordinator system"""
        try:
            logger.info("Initializing Engine Coordinator...")
            
            # Initialize discovery protocol
            self.discovery_protocol = EngineDiscoveryProtocol(self.coordinator_identity)
            await self.discovery_protocol.initialize()
            
            # Initialize message router
            self.message_router = MessageRouter(self.coordinator_identity, self.discovery_protocol)
            
            # Initialize partnership manager
            self.partnership_manager = PartnershipManager(
                self.coordinator_identity,
                self.discovery_protocol,
                self.message_router
            )
            
            # Initialize AI Agent Coordinator
            self.ai_agent_coordinator = AIAgentCoordinator(self.coordinator_identity)
            
            # Register event handlers
            self._register_event_handlers()
            
            logger.info("Engine Coordinator with AI Agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Engine Coordinator: {e}")
            raise
    
    async def start(self):
        """Start the coordinator system"""
        if self._running:
            return
        
        self._running = True
        
        # Start discovery protocol
        await self.discovery_protocol.start()
        
        # Start background monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._auto_management_task = asyncio.create_task(self._auto_management_loop())
        self._ai_decision_task = asyncio.create_task(self._ai_decision_loop())
        
        logger.info("Engine Coordinator started")
    
    async def stop(self):
        """Stop the coordinator system"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        tasks = [self._monitoring_task, self._auto_management_task, self._ai_decision_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop discovery protocol
        if self.discovery_protocol:
            await self.discovery_protocol.stop()
        
        logger.info("Engine Coordinator stopped")
    
    def _register_event_handlers(self):
        """Register handlers for discovery events"""
        from engines.common.engine_discovery import DiscoveryEventType
        
        async def handle_new_engine(event_data):
            """Handle new engine discovery"""
            engine_data = event_data["payload"]
            engine_id = engine_data["engine_id"]
            
            logger.info(f"New engine discovered: {engine_id}")
            
            # Update metrics
            self.system_metrics["engine_discoveries"] += 1
            
            # Auto-establish partnerships with new engines
            if self.partnership_manager:
                await self._auto_establish_partnerships(engine_id, engine_data)
        
        async def handle_engine_shutdown(event_data):
            """Handle engine shutdown"""
            engine_id = event_data["source_engine_id"]
            logger.info(f"Engine shutdown: {engine_id}")
            
            # Update partnerships and routing tables
            if self.partnership_manager and engine_id in self.partnership_manager.partnerships:
                partnership = self.partnership_manager.partnerships[engine_id]
                partnership.status = partnership.PartnershipStatus.SUSPENDED
        
        self.discovery_protocol.register_event_handler(
            DiscoveryEventType.ENGINE_ANNOUNCEMENT,
            handle_new_engine
        )
        self.discovery_protocol.register_event_handler(
            DiscoveryEventType.ENGINE_SHUTDOWN,
            handle_engine_shutdown
        )
    
    async def _auto_establish_partnerships(self, engine_id: str, engine_data: Dict[str, Any]):
        """Automatically establish partnerships with new engines"""
        try:
            # Check if this is a preferred partner based on capabilities
            engine_capabilities = engine_data.get("capabilities", [])
            
            # Determine partnership type based on engine capabilities
            partnership_type = PartnershipType.SECONDARY
            
            # Upgrade to primary for critical engines
            critical_capabilities = [
                "risk_calculation",
                "portfolio_optimization",
                "machine_learning_inference",
                "real_time_streaming"
            ]
            
            if any(cap in engine_capabilities for cap in critical_capabilities):
                partnership_type = PartnershipType.PRIMARY
            
            # Create partnership
            partnership = self.partnership_manager.create_partnership(
                engine_id,
                partnership_type,
                expected_latency_ms=20.0,
                reliability_requirement=0.95
            )
            
            self.system_metrics["partnerships_established"] += 1
            
            logger.info(f"Auto-established {partnership_type.value} partnership with {engine_id}")
            
        except Exception as e:
            logger.error(f"Failed to establish partnership with {engine_id}: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring of system health"""
        while self._running:
            try:
                # Update system status
                await self._update_system_status()
                
                # Update cached data periodically
                if (datetime.now() - self._last_cache_update).seconds > self._cache_ttl_seconds:
                    await self._update_cached_data()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _auto_management_loop(self):
        """Background automatic management tasks"""
        while self._running:
            try:
                # Auto-manage partnerships
                if self.partnership_manager:
                    await self.partnership_manager.auto_manage_partnerships()
                
                # Clean up stale tasks
                await self._cleanup_stale_tasks()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-management loop: {e}")
                await asyncio.sleep(60)
    
    async def _ai_decision_loop(self):
        """Background AI agent decision making"""
        while self._running:
            try:
                if self.ai_agent_coordinator and self.partnership_manager and self.message_router:
                    # Get collective decision from AI agents
                    context = {
                        "system_load": "medium",  # Could be calculated from metrics
                        "market_hours": 9 <= datetime.now().hour <= 16,
                        "total_engines": len(self.discovery_protocol.get_online_engines()) if self.discovery_protocol else 0
                    }
                    
                    decision_result = await self.ai_agent_coordinator.get_collective_decision(
                        self.partnership_manager,
                        self.message_router,
                        context
                    )
                    
                    # Log important decisions
                    consensus = decision_result.get("consensus", {})
                    if consensus.get("consensus_confidence", 0) > 0.7:
                        logger.info(f"AI Agents recommend: {consensus.get('consensus_action', 'no_action')}")
                
                await asyncio.sleep(120)  # AI decisions every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AI decision loop: {e}")
                await asyncio.sleep(180)
    
    async def _update_system_status(self):
        """Update overall system status"""
        try:
            online_engines = self.discovery_protocol.get_online_engines()
            active_partnerships = len(self.partnership_manager.get_active_partnerships()) if self.partnership_manager else 0
            active_tasks = len(self.message_router.active_tasks) if self.message_router else 0
            
            # Determine system health
            total_expected_engines = self.environment.total_engines
            online_ratio = len(online_engines) / max(total_expected_engines, 1)
            
            if online_ratio > 0.9:
                health = "healthy"
            elif online_ratio > 0.7:
                health = "degraded"
            else:
                health = "critical"
            
            # Update metrics
            self.system_metrics.update({
                "total_engines_online": len(online_engines),
                "active_partnerships": active_partnerships,
                "active_tasks": active_tasks,
                "system_health": health,
                "online_ratio": online_ratio,
                "last_status_update": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to update system status: {e}")
    
    async def _update_cached_data(self):
        """Update cached data for API responses"""
        try:
            # Update network topology
            self._cached_network_topology = await self._generate_network_topology()
            
            # Update performance stats
            self._cached_performance_stats = await self._generate_performance_stats()
            
            self._last_cache_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update cached data: {e}")
    
    async def _generate_network_topology(self) -> Dict[str, Any]:
        """Generate network topology data"""
        topology = {
            "nodes": [],
            "edges": [],
            "clusters": {},
            "statistics": {}
        }
        
        online_engines = self.discovery_protocol.get_online_engines()
        
        # Create nodes
        for engine_id, engine_data in online_engines.items():
            node = {
                "id": engine_id,
                "name": engine_data.get("engine_name", engine_id),
                "port": engine_data.get("port", 0),
                "status": engine_data.get("status", "unknown"),
                "roles": engine_data.get("roles", []),
                "capabilities": engine_data.get("capabilities", []),
                "performance": engine_data.get("performance", {}),
                "health": engine_data.get("health", {})
            }
            topology["nodes"].append(node)
        
        # Create edges (partnerships)
        if self.partnership_manager:
            for partner_id, partnership in self.partnership_manager.partnerships.items():
                if partner_id in online_engines:
                    edge = {
                        "source": self.coordinator_identity.engine_id,
                        "target": partner_id,
                        "type": partnership.partnership_type.value,
                        "status": partnership.status.value,
                        "strength": partnership.relationship_strength,
                        "performance_score": partnership.performance_score,
                        "message_count": partnership.message_count
                    }
                    topology["edges"].append(edge)
        
        # Group by roles/capabilities
        role_clusters = {}
        for node in topology["nodes"]:
            for role in node["roles"]:
                if role not in role_clusters:
                    role_clusters[role] = []
                role_clusters[role].append(node["id"])
        
        topology["clusters"] = role_clusters
        topology["statistics"] = {
            "total_nodes": len(topology["nodes"]),
            "total_edges": len(topology["edges"]),
            "cluster_count": len(role_clusters)
        }
        
        return topology
    
    async def _generate_performance_stats(self) -> Dict[str, Any]:
        """Generate performance statistics"""
        stats = {
            "system_metrics": self.system_metrics.copy(),
            "engine_performance": {},
            "partnership_performance": {},
            "routing_performance": {}
        }
        
        # Engine performance
        online_engines = self.discovery_protocol.get_online_engines()
        for engine_id, engine_data in online_engines.items():
            perf = engine_data.get("performance", {})
            health = engine_data.get("health", {})
            
            stats["engine_performance"][engine_id] = {
                "response_time_ms": perf.get("response_time_ms", 0),
                "throughput": perf.get("throughput", 0),
                "acceleration_factor": perf.get("acceleration_factor", 1),
                "status": health.get("status", "unknown"),
                "uptime_seconds": health.get("uptime_seconds", 0),
                "load_pct": health.get("load_pct", 0)
            }
        
        # Partnership performance
        if self.partnership_manager:
            partnership_stats = self.partnership_manager.get_partnership_statistics()
            stats["partnership_performance"] = partnership_stats
        
        # Routing performance
        if self.message_router:
            routing_stats = self.message_router.get_routing_statistics()
            stats["routing_performance"] = routing_stats
        
        return stats
    
    async def _cleanup_stale_tasks(self):
        """Clean up stale tasks"""
        if not self.message_router:
            return
        
        current_time = datetime.now()
        stale_tasks = []
        
        for task_id, task in self.message_router.active_tasks.items():
            task_age = current_time - datetime.fromisoformat(task.created_at.replace('Z', '+00:00'))
            
            # Tasks older than 1 hour are considered stale
            if task_age > timedelta(hours=1):
                stale_tasks.append(task_id)
        
        for task_id in stale_tasks:
            del self.message_router.active_tasks[task_id]
            logger.info(f"Cleaned up stale task: {task_id}")
    
    # API Methods
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        online_engines = self.discovery_protocol.get_online_engines() if self.discovery_protocol else {}
        active_partnerships = len(self.partnership_manager.get_active_partnerships()) if self.partnership_manager else 0
        active_tasks = len(self.message_router.active_tasks) if self.message_router else 0
        
        return SystemStatus(
            total_engines=self.environment.total_engines,
            online_engines=len(online_engines),
            active_partnerships=active_partnerships,
            active_tasks=active_tasks,
            system_health=self.system_metrics.get("system_health", "unknown"),
            last_updated=datetime.now().isoformat()
        )
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get current network topology"""
        return self._cached_network_topology
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return self._cached_performance_stats
    
    def get_engine_details(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific engine"""
        if not self.discovery_protocol:
            return None
        
        engine_data = self.discovery_protocol.registry.get_engine(engine_id)
        if not engine_data:
            return None
        
        details = engine_data.copy()
        
        # Add partnership information
        if self.partnership_manager:
            partnership = self.partnership_manager.get_partnership(engine_id)
            if partnership:
                details["partnership"] = {
                    "type": partnership.partnership_type.value,
                    "status": partnership.status.value,
                    "performance_score": partnership.performance_score,
                    "relationship_strength": partnership.relationship_strength,
                    "message_count": partnership.message_count
                }
        
        return details
    
    async def execute_workflow(self, workflow_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a predefined workflow"""
        if not self.message_router:
            raise HTTPException(status_code=503, detail="Message router not available")
        
        # Get workflow template
        workflow_templates = {
            "portfolio_optimization": WorkflowTemplates.portfolio_optimization_workflow(),
            "market_analysis": WorkflowTemplates.market_analysis_workflow(),
            "risk_monitoring": WorkflowTemplates.risk_monitoring_workflow()
        }
        
        if workflow_name not in workflow_templates:
            raise HTTPException(status_code=400, detail=f"Unknown workflow: {workflow_name}")
        
        workflow = workflow_templates[workflow_name]
        
        # Create and execute distributed task
        task_id = await self.message_router.create_distributed_task(
            workflow_name,
            workflow,
            TaskPriority.HIGH
        )
        
        result = await self.message_router.execute_distributed_task(task_id)
        
        # Update metrics
        if result["success"]:
            self.system_metrics["total_tasks_completed"] += 1
        
        return {
            "task_id": task_id,
            "workflow_name": workflow_name,
            "result": result,
            "executed_at": datetime.now().isoformat()
        }


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    coordinator = EngineCoordinator()
    await coordinator.initialize()
    await coordinator.start()
    
    app.state.coordinator = coordinator
    
    yield
    
    # Shutdown
    await coordinator.stop()


app = FastAPI(
    title="Nautilus Engine Coordinator",
    description="Central coordination service for Nautilus trading engines",
    version="2025.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/system/status", response_model=Dict[str, Any])
async def get_system_status():
    """Get overall system status"""
    coordinator = app.state.coordinator
    status = coordinator.get_system_status()
    return asdict(status)


@app.get("/api/v1/engines", response_model=Dict[str, Any])
async def get_engines():
    """Get all discovered engines"""
    coordinator = app.state.coordinator
    if not coordinator.discovery_protocol:
        raise HTTPException(status_code=503, detail="Discovery protocol not available")
    
    return coordinator.discovery_protocol.get_online_engines()


@app.get("/api/v1/engines/{engine_id}", response_model=Dict[str, Any])
async def get_engine(engine_id: str):
    """Get specific engine details"""
    coordinator = app.state.coordinator
    details = coordinator.get_engine_details(engine_id)
    
    if not details:
        raise HTTPException(status_code=404, detail="Engine not found")
    
    return details


@app.get("/api/v1/network/topology", response_model=Dict[str, Any])
async def get_network_topology():
    """Get network topology"""
    coordinator = app.state.coordinator
    return coordinator.get_network_topology()


@app.get("/api/v1/performance/dashboard", response_model=Dict[str, Any])
async def get_performance_dashboard():
    """Get performance dashboard data"""
    coordinator = app.state.coordinator
    return coordinator.get_performance_dashboard()


@app.post("/api/v1/workflows/{workflow_name}/execute", response_model=Dict[str, Any])
async def execute_workflow(workflow_name: str, parameters: Dict[str, Any] = None):
    """Execute a workflow"""
    coordinator = app.state.coordinator
    
    if parameters is None:
        parameters = {}
    
    return await coordinator.execute_workflow(workflow_name, parameters)


@app.get("/api/v1/partnerships", response_model=Dict[str, Any])
async def get_partnerships():
    """Get all partnerships"""
    coordinator = app.state.coordinator
    if not coordinator.partnership_manager:
        raise HTTPException(status_code=503, detail="Partnership manager not available")
    
    return coordinator.partnership_manager.get_partnership_statistics()


@app.get("/api/v1/partnerships/recommendations", response_model=List[Dict[str, Any]])
async def get_partnership_recommendations():
    """Get partnership recommendations"""
    coordinator = app.state.coordinator
    if not coordinator.partnership_manager:
        raise HTTPException(status_code=503, detail="Partnership manager not available")
    
    recommendations = coordinator.partnership_manager.find_partnership_opportunities()
    return [asdict(rec) for rec in recommendations]


@app.get("/api/v1/ai-agents/status", response_model=Dict[str, Any])
async def get_ai_agents_status():
    """Get AI agents status and performance"""
    coordinator = app.state.coordinator
    if not coordinator.ai_agent_coordinator:
        raise HTTPException(status_code=503, detail="AI agent coordinator not available")
    
    return coordinator.ai_agent_coordinator.get_agent_performance_summary()


@app.post("/api/v1/ai-agents/decision", response_model=Dict[str, Any])
async def get_ai_collective_decision(context: Dict[str, Any] = None):
    """Get collective decision from AI agents"""
    coordinator = app.state.coordinator
    if not coordinator.ai_agent_coordinator:
        raise HTTPException(status_code=503, detail="AI agent coordinator not available")
    
    if not coordinator.partnership_manager or not coordinator.message_router:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    if context is None:
        context = {
            "system_load": "medium",
            "market_hours": 9 <= datetime.now().hour <= 16,
            "requested_by_api": True
        }
    
    return await coordinator.ai_agent_coordinator.get_collective_decision(
        coordinator.partnership_manager,
        coordinator.message_router,
        context
    )


@app.post("/api/v1/ai-agents/strategy", response_model=Dict[str, Any])
async def create_master_strategy(objectives: List[str]):
    """Create master collaboration strategy using AI agents"""
    coordinator = app.state.coordinator
    if not coordinator.ai_agent_coordinator:
        raise HTTPException(status_code=503, detail="AI agent coordinator not available")
    
    if not coordinator.discovery_protocol:
        raise HTTPException(status_code=503, detail="Discovery protocol not available")
    
    available_engines = coordinator.discovery_protocol.get_online_engines()
    
    return await coordinator.ai_agent_coordinator.create_master_strategy(
        available_engines,
        objectives
    )


@app.get("/api/v1/ai-agents/decisions/history", response_model=List[Dict[str, Any]])
async def get_ai_decision_history(limit: int = 20):
    """Get AI agents decision history"""
    coordinator = app.state.coordinator
    if not coordinator.ai_agent_coordinator:
        raise HTTPException(status_code=503, detail="AI agent coordinator not available")
    
    history = coordinator.ai_agent_coordinator.decisions_history[-limit:] if limit > 0 else coordinator.ai_agent_coordinator.decisions_history
    return history


@app.get("/api/v1/system/intelligence", response_model=Dict[str, Any])
async def get_system_intelligence_report():
    """Get comprehensive system intelligence report"""
    coordinator = app.state.coordinator
    
    # System status
    system_status = coordinator.get_system_status()
    
    # AI agents performance
    ai_performance = {}
    if coordinator.ai_agent_coordinator:
        ai_performance = coordinator.ai_agent_coordinator.get_agent_performance_summary()
    
    # Network topology
    topology = coordinator.get_network_topology()
    
    # Partnership stats
    partnership_stats = {}
    if coordinator.partnership_manager:
        partnership_stats = coordinator.partnership_manager.get_partnership_statistics()
    
    return {
        "report_timestamp": datetime.now().isoformat(),
        "system_status": asdict(system_status),
        "ai_agents": ai_performance,
        "network_topology_summary": {
            "total_nodes": len(topology.get("nodes", [])),
            "total_connections": len(topology.get("edges", [])),
            "cluster_count": len(topology.get("clusters", {}))
        },
        "partnership_summary": partnership_stats,
        "intelligence_level": _calculate_system_intelligence_level(
            system_status, ai_performance, partnership_stats
        )
    }


def _calculate_system_intelligence_level(
    system_status: SystemStatus, 
    ai_performance: Dict[str, Any], 
    partnership_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate overall system intelligence level"""
    
    # Base intelligence from system health
    base_score = 0.5
    
    # System availability factor
    if hasattr(system_status, 'online_engines') and hasattr(system_status, 'total_engines'):
        availability_ratio = system_status.online_engines / max(system_status.total_engines, 1)
        base_score += availability_ratio * 0.2
    
    # AI agents factor
    ai_score = 0.0
    if ai_performance.get("total_agents", 0) > 0:
        agent_performance = ai_performance.get("agent_performance", {})
        avg_success_rate = sum(
            agent.get("success_rate", 0) for agent in agent_performance.values()
        ) / len(agent_performance) if agent_performance else 0
        
        ai_score = avg_success_rate * 0.3
    
    # Partnership effectiveness factor
    partnership_score = 0.0
    if partnership_stats:
        avg_performance = partnership_stats.get("average_performance_score", 0)
        partnership_score = avg_performance * 0.2
    
    total_score = min(1.0, base_score + ai_score + partnership_score)
    
    # Determine intelligence level
    if total_score > 0.9:
        level = "genius"
        description = "System demonstrates exceptional collaborative intelligence"
    elif total_score > 0.8:
        level = "advanced"
        description = "System shows advanced learning and adaptation capabilities"
    elif total_score > 0.7:
        level = "intelligent"
        description = "System exhibits intelligent collaborative behaviors"
    elif total_score > 0.6:
        level = "learning"
        description = "System is actively learning and improving"
    else:
        level = "basic"
        description = "System has basic coordination capabilities"
    
    return {
        "level": level,
        "score": total_score,
        "description": description,
        "factors": {
            "system_availability": availability_ratio if 'availability_ratio' in locals() else 0,
            "ai_effectiveness": ai_score / 0.3 if ai_score > 0 else 0,
            "partnership_quality": partnership_score / 0.2 if partnership_score > 0 else 0
        }
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
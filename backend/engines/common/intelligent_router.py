#!/usr/bin/env python3
"""
Intelligent Communication Protocol & Router
Smart routing system that enables engines to collaborate on complex tasks and optimize message flows.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque

from .engine_identity import EngineIdentity, ProcessingCapability
from .engine_discovery import EngineDiscoveryProtocol, EngineRegistry
from .nautilus_environment import get_nautilus_environment, MessageBusType


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task execution priority levels"""
    CRITICAL = "critical"      # Risk alerts, system failures
    HIGH = "high"             # Trading signals, portfolio updates
    MEDIUM = "medium"         # Analytics, feature calculations
    LOW = "low"              # Backtesting, historical analysis


class TaskStatus(Enum):
    """Status of distributed tasks"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoutingStrategy(Enum):
    """Message routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    CAPABILITY_MATCH = "capability_match"
    PARTNERSHIP_PREFERENCE = "partnership_preference"


@dataclass
class TaskStep:
    """Single step in a distributed task workflow"""
    step_id: str
    engine_id: str
    action: str
    input_data: Dict[str, Any]
    output_requirements: List[str]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 2
    status: TaskStatus = TaskStatus.PENDING
    assigned_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DistributedTask:
    """Multi-engine collaborative task"""
    task_id: str
    task_name: str
    requester_engine_id: str
    priority: TaskPriority
    workflow_steps: List[TaskStep]
    created_at: str = ""
    deadline: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class RouteMetrics:
    """Performance metrics for engine routes"""
    engine_id: str
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    average_response_time_ms: float = 0.0
    last_response_time_ms: float = 0.0
    current_load: int = 0
    availability_score: float = 1.0
    partnership_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_messages == 0:
            return 1.0
        return self.successful_messages / self.total_messages
    
    @property
    def overall_score(self) -> float:
        """Combined performance score (0.0-1.0)"""
        response_score = max(0, 1 - (self.average_response_time_ms / 1000))  # Penalize slow responses
        load_score = max(0, 1 - (self.current_load / 100))  # Penalize high load
        
        return (
            self.success_rate * 0.3 +
            response_score * 0.25 +
            load_score * 0.2 +
            self.availability_score * 0.15 +
            self.partnership_score * 0.1
        )


class MessageRouter:
    """Intelligent message routing system"""
    
    def __init__(self, engine_identity: EngineIdentity, discovery_protocol: EngineDiscoveryProtocol):
        self.identity = engine_identity
        self.discovery = discovery_protocol
        
        # Routing metrics and state
        self.route_metrics: Dict[str, RouteMetrics] = {}
        self.message_history: deque = deque(maxlen=1000)
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Task management
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.task_assignments: Dict[str, List[str]] = {}  # engine_id -> [task_ids]
        
        # Performance tracking
        self.performance_window = timedelta(minutes=10)
        self._performance_history: deque = deque(maxlen=1000)
    
    def update_route_metrics(self, engine_id: str, success: bool, response_time_ms: float):
        """Update routing metrics for an engine"""
        if engine_id not in self.route_metrics:
            self.route_metrics[engine_id] = RouteMetrics(engine_id=engine_id)
        
        metrics = self.route_metrics[engine_id]
        metrics.total_messages += 1
        
        if success:
            metrics.successful_messages += 1
        else:
            metrics.failed_messages += 1
        
        # Update response time with exponential moving average
        alpha = 0.1
        if metrics.average_response_time_ms == 0:
            metrics.average_response_time_ms = response_time_ms
        else:
            metrics.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * metrics.average_response_time_ms
            )
        
        metrics.last_response_time_ms = response_time_ms
        
        # Update partnership score based on preferred partners
        preferred_partners = self.identity.get_preferred_partners()
        if engine_id in preferred_partners:
            metrics.partnership_score = 0.8
        else:
            metrics.partnership_score = 0.4
    
    def select_best_engine(
        self, 
        candidates: List[str], 
        strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
        required_capability: Optional[ProcessingCapability] = None
    ) -> Optional[str]:
        """Select best engine from candidates using specified strategy"""
        
        if not candidates:
            return None
        
        # Filter out offline engines
        online_engines = self.discovery.get_online_engines()
        available_candidates = [eid for eid in candidates if eid in online_engines]
        
        if not available_candidates:
            return None
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_candidates)
        
        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_candidates)
        
        elif strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(available_candidates)
        
        elif strategy == RoutingStrategy.PARTNERSHIP_PREFERENCE:
            return self._select_partnership_preference(available_candidates)
        
        elif strategy == RoutingStrategy.CAPABILITY_MATCH:
            return self._select_capability_match(available_candidates, required_capability)
        
        # Default to first available
        return available_candidates[0]
    
    def _select_round_robin(self, candidates: List[str]) -> str:
        """Round-robin selection"""
        key = ",".join(sorted(candidates))
        index = self.round_robin_counters[key] % len(candidates)
        self.round_robin_counters[key] += 1
        return candidates[index]
    
    def _select_least_loaded(self, candidates: List[str]) -> str:
        """Select engine with lowest current load"""
        best_engine = candidates[0]
        best_load = float('inf')
        
        for engine_id in candidates:
            metrics = self.route_metrics.get(engine_id)
            current_load = metrics.current_load if metrics else 0
            
            if current_load < best_load:
                best_load = current_load
                best_engine = engine_id
        
        return best_engine
    
    def _select_fastest_response(self, candidates: List[str]) -> str:
        """Select engine with fastest average response time"""
        best_engine = candidates[0]
        best_time = float('inf')
        
        for engine_id in candidates:
            metrics = self.route_metrics.get(engine_id)
            response_time = metrics.average_response_time_ms if metrics else 1000.0
            
            if response_time < best_time:
                best_time = response_time
                best_engine = engine_id
        
        return best_engine
    
    def _select_partnership_preference(self, candidates: List[str]) -> str:
        """Select based on partnership preferences"""
        preferred_partners = self.identity.get_preferred_partners()
        
        # First priority: preferred partners
        for engine_id in candidates:
            if engine_id in preferred_partners:
                return engine_id
        
        # Second priority: best overall score
        best_engine = candidates[0]
        best_score = 0.0
        
        for engine_id in candidates:
            metrics = self.route_metrics.get(engine_id, RouteMetrics(engine_id=engine_id))
            if metrics.overall_score > best_score:
                best_score = metrics.overall_score
                best_engine = engine_id
        
        return best_engine
    
    def _select_capability_match(self, candidates: List[str], required_capability: Optional[ProcessingCapability]) -> str:
        """Select based on capability match"""
        if not required_capability:
            return self._select_partnership_preference(candidates)
        
        # Find engines with the required capability
        online_engines = self.discovery.get_online_engines()
        capability_matches = []
        
        for engine_id in candidates:
            engine_data = online_engines.get(engine_id, {})
            capabilities = engine_data.get("capabilities", [])
            
            if required_capability.value in capabilities:
                capability_matches.append(engine_id)
        
        if capability_matches:
            return self._select_partnership_preference(capability_matches)
        else:
            return self._select_partnership_preference(candidates)
    
    async def create_distributed_task(
        self,
        task_name: str,
        workflow_steps: List[Dict[str, Any]],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """Create a distributed task workflow"""
        
        task_id = str(uuid.uuid4())
        
        # Convert workflow steps to TaskStep objects
        steps = []
        for i, step_data in enumerate(workflow_steps):
            step = TaskStep(
                step_id=f"{task_id}_step_{i}",
                engine_id=step_data["engine_id"],
                action=step_data["action"],
                input_data=step_data.get("input_data", {}),
                output_requirements=step_data.get("output_requirements", []),
                dependencies=step_data.get("dependencies", []),
                timeout_seconds=step_data.get("timeout_seconds", 30)
            )
            steps.append(step)
        
        task = DistributedTask(
            task_id=task_id,
            task_name=task_name,
            requester_engine_id=self.identity.engine_id,
            priority=priority,
            workflow_steps=steps
        )
        
        self.active_tasks[task_id] = task
        
        logger.info(f"Created distributed task: {task_name} ({task_id})")
        return task_id
    
    async def execute_distributed_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a distributed task workflow"""
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # Execute steps in dependency order
            completed_steps = set()
            
            while task.current_step < len(task.workflow_steps):
                current_step = task.workflow_steps[task.current_step]
                
                # Check if dependencies are satisfied
                if all(dep in completed_steps for dep in current_step.dependencies):
                    
                    # Execute step
                    result = await self._execute_task_step(current_step)
                    
                    if result["success"]:
                        current_step.status = TaskStatus.COMPLETED
                        current_step.completed_at = datetime.now().isoformat()
                        current_step.result = result["data"]
                        completed_steps.add(current_step.step_id)
                        
                        # Store result in task results
                        task.results[current_step.step_id] = result["data"]
                        
                        task.current_step += 1
                        
                    else:
                        # Handle failure
                        current_step.error = result["error"]
                        
                        if current_step.retry_count < current_step.max_retries:
                            current_step.retry_count += 1
                            logger.warning(f"Retrying step {current_step.step_id} ({current_step.retry_count}/{current_step.max_retries})")
                        else:
                            current_step.status = TaskStatus.FAILED
                            task.status = TaskStatus.FAILED
                            logger.error(f"Task step failed: {current_step.step_id}")
                            break
                
                else:
                    # Wait for dependencies
                    await asyncio.sleep(0.1)
            
            if task.status != TaskStatus.FAILED:
                task.status = TaskStatus.COMPLETED
            
            logger.info(f"Distributed task completed: {task.task_name} ({task_id})")
            return {"success": True, "results": task.results}
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            logger.error(f"Error executing distributed task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_task_step(self, step: TaskStep) -> Dict[str, Any]:
        """Execute a single task step"""
        
        step.status = TaskStatus.IN_PROGRESS
        step.assigned_at = datetime.now().isoformat()
        
        try:
            # Here you would send the task to the target engine
            # For now, we'll simulate the execution
            
            logger.info(f"Executing step {step.step_id} on {step.engine_id}: {step.action}")
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Simulate success/failure based on engine availability
            online_engines = self.discovery.get_online_engines()
            if step.engine_id in online_engines:
                # Update routing metrics
                self.update_route_metrics(step.engine_id, True, 100.0)
                
                return {
                    "success": True,
                    "data": {
                        "action": step.action,
                        "processed_by": step.engine_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Engine {step.engine_id} is not available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        stats = {
            "total_engines": len(self.route_metrics),
            "active_tasks": len(self.active_tasks),
            "engine_metrics": {}
        }
        
        for engine_id, metrics in self.route_metrics.items():
            stats["engine_metrics"][engine_id] = {
                "success_rate": metrics.success_rate,
                "average_response_time_ms": metrics.average_response_time_ms,
                "total_messages": metrics.total_messages,
                "current_load": metrics.current_load,
                "overall_score": metrics.overall_score
            }
        
        return stats
    
    def suggest_partnerships(self) -> List[Dict[str, Any]]:
        """Suggest new partnerships based on routing patterns"""
        suggestions = []
        
        # Analyze message patterns to suggest partnerships
        online_engines = self.discovery.get_online_engines()
        
        for engine_id, engine_data in online_engines.items():
            if engine_id == self.identity.engine_id:
                continue
            
            metrics = self.route_metrics.get(engine_id)
            if metrics and metrics.total_messages > 10:
                # Suggest partnership if we communicate frequently
                suggestion = {
                    "engine_id": engine_id,
                    "reason": "frequent_communication",
                    "message_count": metrics.total_messages,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.average_response_time_ms,
                    "suggested_relationship": "secondary"
                }
                
                # Upgrade to primary if very reliable and fast
                if metrics.success_rate > 0.95 and metrics.average_response_time_ms < 50:
                    suggestion["suggested_relationship"] = "primary"
                
                suggestions.append(suggestion)
        
        # Sort by overall score
        suggestions.sort(key=lambda x: x["success_rate"], reverse=True)
        return suggestions[:5]  # Top 5 suggestions


class WorkflowTemplates:
    """Predefined workflow templates for common tasks"""
    
    @staticmethod
    def portfolio_optimization_workflow() -> List[Dict[str, Any]]:
        """Complete portfolio optimization workflow"""
        return [
            {
                "engine_id": "FACTOR_ENGINE",
                "action": "calculate_factors",
                "input_data": {"universe": "SP500"},
                "output_requirements": ["factor_scores"],
                "dependencies": []
            },
            {
                "engine_id": "FEATURES_ENGINE",
                "action": "engineer_features",
                "input_data": {"factors": "previous_step"},
                "output_requirements": ["feature_matrix"],
                "dependencies": ["portfolio_optimization_workflow_step_0"]
            },
            {
                "engine_id": "ML_ENGINE",
                "action": "predict_returns",
                "input_data": {"features": "previous_step"},
                "output_requirements": ["return_predictions"],
                "dependencies": ["portfolio_optimization_workflow_step_1"]
            },
            {
                "engine_id": "RISK_ENGINE",
                "action": "assess_risk",
                "input_data": {"predictions": "previous_step"},
                "output_requirements": ["risk_metrics"],
                "dependencies": ["portfolio_optimization_workflow_step_2"]
            },
            {
                "engine_id": "QUANTUM_PORTFOLIO_ENGINE",
                "action": "quantum_optimize",
                "input_data": {"returns": "step_2", "risk": "step_3"},
                "output_requirements": ["optimal_weights"],
                "dependencies": ["portfolio_optimization_workflow_step_2", "portfolio_optimization_workflow_step_3"]
            },
            {
                "engine_id": "COLLATERAL_ENGINE",
                "action": "verify_margins",
                "input_data": {"weights": "previous_step"},
                "output_requirements": ["margin_requirements"],
                "dependencies": ["portfolio_optimization_workflow_step_4"]
            }
        ]
    
    @staticmethod
    def market_analysis_workflow() -> List[Dict[str, Any]]:
        """Real-time market analysis workflow"""
        return [
            {
                "engine_id": "IBKR_KEEPALIVE_ENGINE",
                "action": "fetch_market_data",
                "input_data": {"symbols": ["SPY", "QQQ", "IWM"]},
                "output_requirements": ["market_data"],
                "dependencies": []
            },
            {
                "engine_id": "VPIN_ENGINE",
                "action": "calculate_vpin",
                "input_data": {"market_data": "previous_step"},
                "output_requirements": ["vpin_scores"],
                "dependencies": ["market_analysis_workflow_step_0"]
            },
            {
                "engine_id": "WEBSOCKET_THGNN_ENGINE",
                "action": "hft_prediction",
                "input_data": {"market_data": "step_0", "vpin": "step_1"},
                "output_requirements": ["hft_signals"],
                "dependencies": ["market_analysis_workflow_step_0", "market_analysis_workflow_step_1"]
            },
            {
                "engine_id": "STRATEGY_ENGINE",
                "action": "generate_signals",
                "input_data": {"hft_signals": "previous_step"},
                "output_requirements": ["trading_signals"],
                "dependencies": ["market_analysis_workflow_step_2"]
            }
        ]
    
    @staticmethod
    def risk_monitoring_workflow() -> List[Dict[str, Any]]:
        """Continuous risk monitoring workflow"""
        return [
            {
                "engine_id": "PORTFOLIO_ENGINE",
                "action": "get_positions",
                "input_data": {},
                "output_requirements": ["current_positions"],
                "dependencies": []
            },
            {
                "engine_id": "RISK_ENGINE",
                "action": "calculate_var",
                "input_data": {"positions": "previous_step"},
                "output_requirements": ["var_metrics"],
                "dependencies": ["risk_monitoring_workflow_step_0"]
            },
            {
                "engine_id": "COLLATERAL_ENGINE",
                "action": "monitor_margins",
                "input_data": {"positions": "step_0", "risk": "step_1"},
                "output_requirements": ["margin_status"],
                "dependencies": ["risk_monitoring_workflow_step_0", "risk_monitoring_workflow_step_1"]
            }
        ]


if __name__ == "__main__":
    # Demo usage
    from .engine_identity import create_ml_engine_identity
    
    async def demo():
        ml_engine = create_ml_engine_identity()
        discovery = EngineDiscoveryProtocol(ml_engine)
        router = MessageRouter(ml_engine, discovery)
        
        print("=== Intelligent Router Demo ===")
        
        # Create portfolio optimization task
        workflow = WorkflowTemplates.portfolio_optimization_workflow()
        task_id = await router.create_distributed_task(
            "Portfolio Optimization",
            workflow,
            TaskPriority.HIGH
        )
        
        print(f"Created task: {task_id}")
        
        # Get routing statistics
        stats = router.get_routing_statistics()
        print(f"Routing stats: {json.dumps(stats, indent=2)}")
        
        # Get partnership suggestions
        suggestions = router.suggest_partnerships()
        print(f"Partnership suggestions: {len(suggestions)}")
    
    asyncio.run(demo())
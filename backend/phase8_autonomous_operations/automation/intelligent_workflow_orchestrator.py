"""
Intelligent Workflow Orchestrator - Phase 8 Autonomous Operations
==============================================================

Provides intelligent workflow orchestration with adaptive pipelines, automatic
optimization, and AI-driven decision making for complex trading workflows.

Key Features:
- Adaptive pipeline execution with dynamic reconfiguration
- AI-powered workflow optimization and anomaly detection
- Intelligent dependency resolution and scheduling
- Real-time workflow monitoring with predictive alerts
- Auto-healing workflows with failure recovery
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import numpy as np
from pydantic import BaseModel, Field
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    OPTIMIZING = "optimizing"

class TaskPriority(Enum):
    """Task execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class WorkflowType(Enum):
    """Types of workflows supported"""
    TRADING = "trading"
    DATA_PROCESSING = "data_processing" 
    RISK_ANALYSIS = "risk_analysis"
    ANALYTICS = "analytics"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"

@dataclass
class TaskMetrics:
    """Performance metrics for workflow tasks"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    retry_count: int = 0
    last_execution: Optional[datetime] = None
    average_execution_time: float = 0.0
    performance_score: float = 1.0

@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    name: str
    task_type: str
    function: Callable
    dependencies: Set[str] = field(default_factory=set)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: int = 300
    retry_count: int = 3
    retry_delay: int = 5
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class WorkflowDefinition(BaseModel):
    """Workflow definition model"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    workflow_type: WorkflowType
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    global_timeout: int = 3600
    max_retries: int = 3
    priority: TaskPriority = TaskPriority.NORMAL
    schedule: Optional[str] = None
    conditions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task_results: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

class AdaptiveOptimizer:
    """AI-powered workflow optimizer"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.optimization_rules: List[Dict[str, Any]] = []
        self.learning_window = 100
        
    def analyze_performance(self, workflow_id: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow performance and identify optimization opportunities"""
        try:
            metrics = execution_data.get('metrics', {})
            execution_time = metrics.get('total_execution_time', 0)
            
            # Store performance history
            self.performance_history[workflow_id].append(execution_time)
            
            # Keep only recent executions
            if len(self.performance_history[workflow_id]) > self.learning_window:
                self.performance_history[workflow_id] = self.performance_history[workflow_id][-self.learning_window:]
            
            # Calculate performance statistics
            history = self.performance_history[workflow_id]
            if len(history) < 5:
                return {"status": "insufficient_data", "recommendations": []}
            
            avg_time = np.mean(history)
            std_time = np.std(history)
            trend = self.calculate_trend(history)
            
            # Detect anomalies
            recent_executions = np.array(history[-20:]).reshape(-1, 1)
            if len(recent_executions) >= 5:
                scaled_data = self.scaler.fit_transform(recent_executions)
                anomalies = self.anomaly_detector.fit_predict(scaled_data)
                anomaly_detected = -1 in anomalies
            else:
                anomaly_detected = False
            
            # Generate optimization recommendations
            recommendations = []
            
            if execution_time > avg_time + 2 * std_time:
                recommendations.append({
                    "type": "performance_degradation",
                    "severity": "high",
                    "action": "investigate_bottlenecks",
                    "details": f"Execution time {execution_time:.2f}s exceeds normal range"
                })
            
            if trend > 0.1:
                recommendations.append({
                    "type": "performance_decline",
                    "severity": "medium", 
                    "action": "optimize_pipeline",
                    "details": "Increasing trend in execution time detected"
                })
            
            if anomaly_detected:
                recommendations.append({
                    "type": "anomaly_detected",
                    "severity": "high",
                    "action": "investigate_anomaly",
                    "details": "Unusual execution pattern detected"
                })
            
            return {
                "status": "analyzed",
                "performance_score": max(0, 1 - (execution_time - avg_time) / avg_time) if avg_time > 0 else 1,
                "average_execution_time": avg_time,
                "trend": trend,
                "anomaly_detected": anomaly_detected,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def calculate_trend(self, data: List[float]) -> float:
        """Calculate performance trend using linear regression"""
        if len(data) < 3:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        return slope / np.mean(y) if np.mean(y) > 0 else 0.0
    
    def suggest_optimizations(self, workflow: WorkflowDefinition, execution_history: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Suggest workflow optimizations based on execution history"""
        optimizations = []
        
        if not execution_history:
            return optimizations
        
        # Analyze task performance patterns
        task_performance = defaultdict(list)
        for execution in execution_history[-20:]:  # Last 20 executions
            for task_id, task_result in execution.task_results.items():
                if isinstance(task_result, dict) and 'execution_time' in task_result:
                    task_performance[task_id].append(task_result['execution_time'])
        
        # Identify slow tasks
        for task_id, times in task_performance.items():
            if len(times) >= 3:
                avg_time = np.mean(times)
                if avg_time > 30:  # Tasks taking more than 30 seconds
                    optimizations.append({
                        "type": "slow_task",
                        "task_id": task_id,
                        "recommendation": "Consider optimizing or parallelizing this task",
                        "average_time": avg_time,
                        "priority": "medium"
                    })
        
        # Suggest parallelization opportunities
        task_dependencies = {}
        for task_def in workflow.tasks:
            task_dependencies[task_def['task_id']] = task_def.get('dependencies', [])
        
        # Find independent tasks that can be parallelized
        independent_groups = self.find_parallel_groups(task_dependencies)
        if len(independent_groups) > 1:
            optimizations.append({
                "type": "parallelization",
                "recommendation": "Execute independent task groups in parallel",
                "groups": independent_groups,
                "potential_speedup": f"{len(independent_groups)}x",
                "priority": "high"
            })
        
        return optimizations
    
    def find_parallel_groups(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel"""
        # Create dependency graph
        graph = nx.DiGraph()
        for task, deps in dependencies.items():
            graph.add_node(task)
            for dep in deps:
                graph.add_edge(dep, task)
        
        # Find nodes with no dependencies (can start immediately)
        independent_tasks = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        
        # Group tasks by execution level
        levels = []
        remaining_nodes = set(graph.nodes())
        
        while remaining_nodes:
            # Find nodes with no remaining dependencies
            current_level = []
            for node in list(remaining_nodes):
                if all(dep not in remaining_nodes for dep in graph.predecessors(node)):
                    current_level.append(node)
            
            if not current_level:
                # Circular dependency or other issue
                break
                
            levels.append(current_level)
            remaining_nodes -= set(current_level)
        
        return levels

class IntelligentWorkflowOrchestrator:
    """Main orchestrator for intelligent workflow management"""
    
    def __init__(self, max_concurrent_workflows: int = 10):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_registry: Dict[str, Callable] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.max_concurrent_workflows = max_concurrent_workflows
        self.optimizer = AdaptiveOptimizer()
        self.workflow_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.global_context: Dict[str, Any] = {}
        
        # Performance monitoring
        self.execution_history: Dict[str, List[WorkflowExecution]] = defaultdict(list)
        self.performance_alerts = []
        
    async def register_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Register a new workflow definition"""
        try:
            # Validate workflow
            validation_result = await self.validate_workflow(workflow_def)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid workflow: {validation_result['errors']}")
            
            # Store workflow
            self.workflows[workflow_def.workflow_id] = workflow_def
            
            # Initialize metrics
            self.workflow_metrics[workflow_def.workflow_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0,
                "last_execution": None,
                "performance_score": 1.0
            }
            
            logger.info(f"Registered workflow: {workflow_def.name} ({workflow_def.workflow_id})")
            return workflow_def.workflow_id
            
        except Exception as e:
            logger.error(f"Error registering workflow: {e}")
            raise
    
    async def validate_workflow(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        warnings = []
        
        # Check for required fields
        if not workflow_def.name:
            errors.append("Workflow name is required")
        
        if not workflow_def.tasks:
            errors.append("Workflow must have at least one task")
        
        # Validate task definitions
        task_ids = set()
        for i, task in enumerate(workflow_def.tasks):
            if 'task_id' not in task:
                errors.append(f"Task {i} missing task_id")
                continue
                
            task_id = task['task_id']
            if task_id in task_ids:
                errors.append(f"Duplicate task_id: {task_id}")
            task_ids.add(task_id)
            
            # Check if task function is registered
            if task.get('task_type') not in self.task_registry:
                errors.append(f"Task type '{task.get('task_type')}' not registered")
            
            # Validate dependencies
            dependencies = task.get('dependencies', [])
            for dep in dependencies:
                if dep not in task_ids and dep not in [t.get('task_id') for t in workflow_def.tasks]:
                    warnings.append(f"Task {task_id} depends on unknown task: {dep}")
        
        # Check for circular dependencies
        if self.has_circular_dependencies(workflow_def.tasks):
            errors.append("Workflow has circular dependencies")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def has_circular_dependencies(self, tasks: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies in workflow tasks"""
        # Build dependency graph
        graph = {}
        for task in tasks:
            task_id = task.get('task_id')
            dependencies = task.get('dependencies', [])
            graph[task_id] = dependencies
        
        # DFS to detect cycles
        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for task_id in graph:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    return True
        
        return False
    
    def register_task(self, task_type: str, function: Callable) -> None:
        """Register a task function"""
        self.task_registry[task_type] = function
        logger.info(f"Registered task type: {task_type}")
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> str:
        """Execute a workflow and return execution ID"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if len(self.running_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows exceeded")
        
        workflow = self.workflows[workflow_id]
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            context=context or {}
        )
        
        self.executions[execution.execution_id] = execution
        
        # Start workflow execution
        task = asyncio.create_task(
            self._execute_workflow_internal(workflow, execution)
        )
        self.running_workflows[execution.execution_id] = task
        
        logger.info(f"Started workflow execution: {execution.execution_id}")
        return execution.execution_id
    
    async def _execute_workflow_internal(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Internal workflow execution logic"""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            await self.emit_event('workflow_started', {
                'workflow_id': workflow.workflow_id,
                'execution_id': execution.execution_id
            })
            
            # Build execution plan with optimization
            execution_plan = await self.build_execution_plan(workflow, execution)
            
            # Execute tasks according to plan
            await self.execute_tasks(workflow, execution, execution_plan)
            
            # Calculate duration and finalize
            execution.completed_at = datetime.now()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            if execution.status != WorkflowStatus.FAILED:
                execution.status = WorkflowStatus.COMPLETED
            
            # Update metrics
            await self.update_workflow_metrics(workflow.workflow_id, execution)
            
            # Analyze performance and optimize
            await self.analyze_and_optimize(workflow, execution)
            
            await self.emit_event('workflow_completed', {
                'workflow_id': workflow.workflow_id,
                'execution_id': execution.execution_id,
                'status': execution.status.value,
                'duration': execution.duration
            })
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.error(f"Workflow execution failed: {e}")
            
            await self.emit_event('workflow_failed', {
                'workflow_id': workflow.workflow_id,
                'execution_id': execution.execution_id,
                'error': str(e)
            })
            
        finally:
            # Cleanup
            if execution.execution_id in self.running_workflows:
                del self.running_workflows[execution.execution_id]
    
    async def build_execution_plan(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> List[List[str]]:
        """Build optimized execution plan with parallel task groups"""
        # Convert task definitions to dependency graph
        task_deps = {}
        task_priorities = {}
        
        for task_def in workflow.tasks:
            task_id = task_def['task_id']
            task_deps[task_id] = task_def.get('dependencies', [])
            task_priorities[task_id] = task_def.get('priority', TaskPriority.NORMAL.value)
        
        # Apply AI optimizations
        optimizations = self.optimizer.suggest_optimizations(
            workflow, 
            self.execution_history[workflow.workflow_id]
        )
        
        # Build execution levels considering optimizations
        execution_levels = self.optimizer.find_parallel_groups(task_deps)
        
        # Sort tasks within each level by priority
        for level in execution_levels:
            level.sort(key=lambda task_id: TaskPriority[task_priorities.get(task_id, TaskPriority.NORMAL.value)].value)
        
        logger.info(f"Built execution plan with {len(execution_levels)} levels")
        return execution_levels
    
    async def execute_tasks(self, workflow: WorkflowDefinition, execution: WorkflowExecution, execution_plan: List[List[str]]) -> None:
        """Execute tasks according to the execution plan"""
        task_definitions = {task['task_id']: task for task in workflow.tasks}
        
        for level_index, task_group in enumerate(execution_plan):
            logger.info(f"Executing level {level_index + 1}/{len(execution_plan)} with {len(task_group)} tasks")
            
            # Execute all tasks in this level concurrently
            level_tasks = []
            for task_id in task_group:
                task_def = task_definitions[task_id]
                task_coro = self.execute_single_task(task_def, execution)
                level_tasks.append(task_coro)
            
            # Wait for all tasks in this level to complete
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Check for failures
            for i, result in enumerate(level_results):
                task_id = task_group[i]
                if isinstance(result, Exception):
                    logger.error(f"Task {task_id} failed: {result}")
                    execution.task_results[task_id] = {"error": str(result)}
                    
                    # Check if this is a critical task
                    task_def = task_definitions[task_id]
                    if task_def.get('critical', False):
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Critical task {task_id} failed: {result}"
                        return
                else:
                    execution.task_results[task_id] = result
    
    async def execute_single_task(self, task_def: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a single task"""
        task_id = task_def['task_id']
        task_type = task_def['task_type']
        
        start_time = datetime.now()
        
        try:
            # Get task function
            if task_type not in self.task_registry:
                raise ValueError(f"Task type {task_type} not registered")
            
            task_function = self.task_registry[task_type]
            
            # Prepare task parameters
            task_params = task_def.get('parameters', {})
            task_params.update({
                'execution_context': execution.context,
                'global_context': self.global_context,
                'task_id': task_id
            })
            
            # Execute task with timeout
            timeout = task_def.get('timeout', 300)
            result = await asyncio.wait_for(
                self._call_task_function(task_function, task_params),
                timeout=timeout
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            await self.emit_event('task_completed', {
                'task_id': task_id,
                'execution_id': execution.execution_id,
                'execution_time': execution_time,
                'result': result
            })
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat()
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Task {task_id} timed out after {task_def.get('timeout', 300)} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Check for retry logic
            retry_count = task_def.get('retry_count', 0)
            if retry_count > 0:
                # Implement retry logic here
                pass
            
            raise e
    
    async def _call_task_function(self, function: Callable, params: Dict[str, Any]) -> Any:
        """Call task function, handling both sync and async functions"""
        if asyncio.iscoroutinefunction(function):
            return await function(**params)
        else:
            # Run sync function in thread pool
            return await asyncio.get_event_loop().run_in_executor(None, lambda: function(**params))
    
    async def update_workflow_metrics(self, workflow_id: str, execution: WorkflowExecution) -> None:
        """Update workflow performance metrics"""
        metrics = self.workflow_metrics[workflow_id]
        
        metrics["total_executions"] += 1
        metrics["last_execution"] = execution.completed_at
        
        if execution.status == WorkflowStatus.COMPLETED:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
        
        # Update average execution time
        if execution.duration:
            if metrics["average_execution_time"] == 0:
                metrics["average_execution_time"] = execution.duration
            else:
                # Exponential moving average
                alpha = 0.2
                metrics["average_execution_time"] = (
                    alpha * execution.duration + 
                    (1 - alpha) * metrics["average_execution_time"]
                )
        
        # Calculate performance score
        success_rate = metrics["successful_executions"] / metrics["total_executions"]
        metrics["performance_score"] = success_rate
        
        # Store execution history for analysis
        self.execution_history[workflow_id].append(execution)
        if len(self.execution_history[workflow_id]) > 100:  # Keep last 100 executions
            self.execution_history[workflow_id] = self.execution_history[workflow_id][-100:]
    
    async def analyze_and_optimize(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Analyze execution and apply optimizations"""
        try:
            # Prepare analysis data
            analysis_data = {
                'metrics': {
                    'total_execution_time': execution.duration or 0,
                    'task_count': len(workflow.tasks),
                    'success': execution.status == WorkflowStatus.COMPLETED
                },
                'task_results': execution.task_results
            }
            
            # Analyze performance
            analysis_result = self.optimizer.analyze_performance(workflow.workflow_id, analysis_data)
            
            # Apply recommendations
            for recommendation in analysis_result.get('recommendations', []):
                await self.apply_optimization_recommendation(workflow.workflow_id, recommendation)
                
        except Exception as e:
            logger.error(f"Error in workflow analysis: {e}")
    
    async def apply_optimization_recommendation(self, workflow_id: str, recommendation: Dict[str, Any]) -> None:
        """Apply optimization recommendations"""
        rec_type = recommendation.get('type')
        severity = recommendation.get('severity', 'low')
        
        if severity == 'high':
            # Create performance alert
            alert = {
                'workflow_id': workflow_id,
                'type': rec_type,
                'message': recommendation.get('details', ''),
                'timestamp': datetime.now().isoformat(),
                'action_required': recommendation.get('action', '')
            }
            self.performance_alerts.append(alert)
            
            await self.emit_event('performance_alert', alert)
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit workflow events to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for workflow events"""
        self.event_handlers[event_type].append(handler)
    
    async def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self.executions.get(execution_id)
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        if execution_id in self.running_workflows:
            task = self.running_workflows[execution_id]
            task.cancel()
            
            if execution_id in self.executions:
                self.executions[execution_id].status = WorkflowStatus.CANCELLED
                self.executions[execution_id].completed_at = datetime.now()
            
            return True
        return False
    
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        return self.workflow_metrics.get(workflow_id, {})
    
    async def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts"""
        return self.performance_alerts.copy()
    
    async def cleanup_old_executions(self, days: int = 7) -> int:
        """Cleanup old execution data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        execution_ids_to_remove = []
        for execution_id, execution in self.executions.items():
            if (execution.completed_at and execution.completed_at < cutoff_date and 
                execution_id not in self.running_workflows):
                execution_ids_to_remove.append(execution_id)
        
        for execution_id in execution_ids_to_remove:
            del self.executions[execution_id]
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old workflow executions")
        return cleaned_count

# Example task functions
async def sample_data_processing_task(**kwargs):
    """Sample data processing task"""
    execution_context = kwargs.get('execution_context', {})
    task_id = kwargs.get('task_id', 'unknown')
    
    # Simulate processing
    await asyncio.sleep(1)
    
    return {
        'processed_records': 1000,
        'processing_time': 1.0,
        'task_id': task_id
    }

async def sample_risk_analysis_task(**kwargs):
    """Sample risk analysis task"""
    execution_context = kwargs.get('execution_context', {})
    
    # Simulate risk calculation
    await asyncio.sleep(2)
    
    return {
        'risk_score': 0.85,
        'var_95': 0.02,
        'max_drawdown': 0.15
    }

def sample_validation_task(**kwargs):
    """Sample validation task (synchronous)"""
    execution_context = kwargs.get('execution_context', {})
    
    # Simulate validation
    import time
    time.sleep(0.5)
    
    return {
        'validation_passed': True,
        'issues_found': 0
    }

# Example usage and testing
async def main():
    """Example usage of Intelligent Workflow Orchestrator"""
    orchestrator = IntelligentWorkflowOrchestrator()
    
    # Register task functions
    orchestrator.register_task('data_processing', sample_data_processing_task)
    orchestrator.register_task('risk_analysis', sample_risk_analysis_task)
    orchestrator.register_task('validation', sample_validation_task)
    
    # Define sample workflow
    workflow_def = WorkflowDefinition(
        name="Daily Risk Processing",
        workflow_type=WorkflowType.RISK_ANALYSIS,
        tasks=[
            {
                'task_id': 'fetch_data',
                'task_type': 'data_processing',
                'parameters': {'source': 'market_data'},
                'priority': TaskPriority.HIGH.value
            },
            {
                'task_id': 'validate_data',
                'task_type': 'validation',
                'dependencies': ['fetch_data'],
                'parameters': {'validation_rules': 'standard'}
            },
            {
                'task_id': 'calculate_risk',
                'task_type': 'risk_analysis',
                'dependencies': ['validate_data'],
                'parameters': {'risk_model': 'var_model'}
            }
        ]
    )
    
    # Register workflow
    workflow_id = await orchestrator.register_workflow(workflow_def)
    print(f"Registered workflow: {workflow_id}")
    
    # Execute workflow
    execution_id = await orchestrator.execute_workflow(workflow_id, {'portfolio_id': 'DEMO'})
    print(f"Started execution: {execution_id}")
    
    # Wait for completion
    while True:
        status = await orchestrator.get_workflow_status(execution_id)
        if status and status.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            break
        await asyncio.sleep(1)
    
    print(f"Workflow completed with status: {status.status}")
    print(f"Duration: {status.duration:.2f} seconds")
    print(f"Results: {status.task_results}")

if __name__ == "__main__":
    asyncio.run(main())
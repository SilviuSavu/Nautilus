"""
Edge Failover Manager for Consistent Trading Operations

This module manages automatic failover and data consistency across edge nodes
to ensure continuous trading operations with minimal disruption.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import hashlib


class FailoverTrigger(Enum):
    """Failover trigger conditions"""
    NODE_FAILURE = "node_failure"                    # Complete node failure
    NETWORK_PARTITION = "network_partition"          # Network connectivity loss
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Performance below thresholds
    PLANNED_MAINTENANCE = "planned_maintenance"      # Scheduled maintenance
    RESOURCE_EXHAUSTION = "resource_exhaustion"      # Resource limits exceeded
    DATA_CORRUPTION = "data_corruption"              # Data integrity issues
    MANUAL_TRIGGER = "manual_trigger"                # Manual operator intervention


class FailoverStrategy(Enum):
    """Failover execution strategies"""
    IMMEDIATE = "immediate"              # Immediate failover without delay
    GRACEFUL = "graceful"               # Graceful failover with connection draining
    ROLLING = "rolling"                 # Rolling failover across multiple nodes
    BLUE_GREEN = "blue_green"           # Blue-green deployment style failover
    CANARY = "canary"                   # Canary-style gradual traffic shift


class ConsistencyModel(Enum):
    """Data consistency models for failover"""
    STRONG_CONSISTENCY = "strong"       # All replicas must be consistent
    EVENTUAL_CONSISTENCY = "eventual"   # Eventually consistent across nodes
    CAUSAL_CONSISTENCY = "causal"       # Causally consistent operations
    MONOTONIC_READ = "monotonic_read"   # Monotonic read consistency
    WEAK_CONSISTENCY = "weak"           # No consistency guarantees


class NodeStatus(Enum):
    """Edge node status states"""
    ACTIVE = "active"                   # Fully operational
    STANDBY = "standby"                 # Ready for failover
    DRAINING = "draining"               # Draining connections before failover
    FAILED = "failed"                   # Node has failed
    MAINTENANCE = "maintenance"         # Under maintenance
    RECOVERING = "recovering"           # Recovering from failure


@dataclass
class FailoverConfiguration:
    """Failover configuration for edge deployment"""
    config_id: str
    deployment_name: str
    
    # Health check configuration
    health_check_interval_seconds: int = 5
    health_check_timeout_seconds: int = 2
    consecutive_failures_threshold: int = 3
    
    # Performance thresholds
    max_latency_us: float = 2000.0
    min_throughput_ops: float = 1000.0
    max_error_rate: float = 0.01  # 1%
    max_cpu_utilization: float = 90.0
    
    # Failover timing
    failover_timeout_seconds: int = 30
    connection_drain_timeout_seconds: int = 60
    recovery_check_interval_seconds: int = 30
    
    # Consistency requirements
    consistency_model: ConsistencyModel = ConsistencyModel.EVENTUAL_CONSISTENCY
    max_data_lag_seconds: float = 5.0
    required_replica_count: int = 2
    
    # Notification settings
    alert_webhook_url: Optional[str] = None
    alert_email_addresses: List[str] = field(default_factory=list)
    alert_slack_channel: Optional[str] = None


@dataclass
class EdgeNodeHealth:
    """Health status of an edge node"""
    node_id: str
    status: NodeStatus
    last_check_time: float
    
    # Health metrics
    latency_us: float
    throughput_ops: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    
    # Connection status
    active_connections: int
    pending_connections: int
    failed_connections: int
    
    # Data consistency status
    last_data_sync_time: float
    data_lag_seconds: float
    pending_sync_operations: int
    
    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class FailoverEvent:
    """Record of failover event"""
    event_id: str
    trigger: FailoverTrigger
    strategy: FailoverStrategy
    
    # Timing
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    
    # Nodes involved
    failed_nodes: List[str]
    target_nodes: List[str]
    affected_connections: int
    
    # Execution details
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    rollback_required: bool = False
    
    # Impact assessment
    data_loss_risk: str = "none"  # none, minimal, moderate, significant
    downtime_seconds: float = 0.0
    affected_trading_sessions: List[str] = field(default_factory=list)
    
    # Recovery status
    recovery_status: str = "pending"  # pending, in_progress, completed, failed
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class DataConsistencyState:
    """Data consistency state across edge nodes"""
    consistency_id: str
    timestamp: float
    
    # Version vectors for consistency tracking
    node_versions: Dict[str, int]  # node_id -> version
    pending_operations: Dict[str, List[str]]  # node_id -> [operations]
    
    # Consistency metrics
    max_version_lag: int
    nodes_out_of_sync: List[str]
    critical_data_inconsistent: bool
    
    # Synchronization status
    sync_in_progress: bool
    sync_target_nodes: List[str]
    estimated_sync_completion_time: float


class EdgeFailoverManager:
    """
    Edge Failover Manager for Consistent Trading Operations
    
    Provides automatic failover capabilities with:
    - Continuous health monitoring across edge nodes
    - Multiple failover strategies and triggers
    - Data consistency guarantees during failover
    - Automatic recovery and rollback procedures
    - Comprehensive event logging and alerting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration and state
        self.failover_configs: Dict[str, FailoverConfiguration] = {}
        self.node_health: Dict[str, EdgeNodeHealth] = {}
        self.failover_events: List[FailoverEvent] = []
        
        # Data consistency tracking
        self.consistency_state: Dict[str, DataConsistencyState] = {}
        self.consistency_checks: Dict[str, asyncio.Task] = {}
        
        # Active failovers
        self.active_failovers: Dict[str, FailoverEvent] = {}
        
        # Monitoring and control
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Health check cache for performance
        self.health_check_cache: Dict[str, Tuple[float, bool]] = {}  # node -> (timestamp, healthy)
        
        self.logger.info("Edge Failover Manager initialized")
    
    async def configure_failover(self, config: FailoverConfiguration):
        """Configure failover settings for edge deployment"""
        
        self.failover_configs[config.config_id] = config
        
        # Initialize consistency state tracking
        self.consistency_state[config.deployment_name] = DataConsistencyState(
            consistency_id=f"consistency_{config.deployment_name}",
            timestamp=time.time(),
            node_versions={},
            pending_operations={},
            max_version_lag=0,
            nodes_out_of_sync=[],
            critical_data_inconsistent=False,
            sync_in_progress=False,
            sync_target_nodes=[],
            estimated_sync_completion_time=0.0
        )
        
        self.logger.info(f"Configured failover for deployment: {config.deployment_name}")
        self.logger.info(f"Health check interval: {config.health_check_interval_seconds}s")
        self.logger.info(f"Consistency model: {config.consistency_model.value}")
    
    async def register_edge_node(
        self, 
        node_id: str, 
        deployment_name: str,
        initial_status: NodeStatus = NodeStatus.STANDBY
    ):
        """Register edge node for failover management"""
        
        self.node_health[node_id] = EdgeNodeHealth(
            node_id=node_id,
            status=initial_status,
            last_check_time=time.time(),
            latency_us=0.0,
            throughput_ops=0.0,
            error_rate=0.0,
            cpu_utilization=0.0,
            memory_utilization=0.0,
            network_utilization=0.0,
            active_connections=0,
            pending_connections=0,
            failed_connections=0,
            last_data_sync_time=time.time(),
            data_lag_seconds=0.0,
            pending_sync_operations=0
        )
        
        # Initialize in consistency state
        if deployment_name in self.consistency_state:
            self.consistency_state[deployment_name].node_versions[node_id] = 0
            self.consistency_state[deployment_name].pending_operations[node_id] = []
        
        self.logger.info(f"Registered edge node: {node_id} (deployment: {deployment_name})")
    
    async def start_monitoring(self):
        """Start continuous health monitoring and failover detection"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._consistency_monitoring_loop()),
            asyncio.create_task(self._recovery_monitoring_loop()),
            asyncio.create_task(self._cleanup_monitoring_loop())
        ]
        
        self.logger.info("Edge failover monitoring started")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Check health of all registered nodes
                health_check_tasks = []
                
                for node_id in self.node_health.keys():
                    task = asyncio.create_task(self._check_node_health(node_id))
                    health_check_tasks.append((node_id, task))
                
                # Wait for all health checks with timeout
                for node_id, task in health_check_tasks:
                    try:
                        await asyncio.wait_for(task, timeout=10.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Health check timeout for node: {node_id}")
                        await self._handle_health_check_failure(node_id, "health_check_timeout")
                    except Exception as e:
                        self.logger.error(f"Health check error for node {node_id}: {e}")
                        await self._handle_health_check_failure(node_id, str(e))
                
                # Evaluate failover conditions
                await self._evaluate_failover_conditions()
                
                # Sleep based on minimum configured interval
                min_interval = min(
                    (config.health_check_interval_seconds for config in self.failover_configs.values()),
                    default=5
                )
                await asyncio.sleep(min_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_node_health(self, node_id: str):
        """Check health of specific edge node"""
        
        try:
            node_health = self.node_health[node_id]
            current_time = time.time()
            
            # Check cache first for performance
            if node_id in self.health_check_cache:
                cached_time, cached_healthy = self.health_check_cache[node_id]
                if current_time - cached_time < 1.0:  # Use cache for 1 second
                    return cached_healthy
            
            # Simulate health check measurements
            health_result = await self._perform_health_check(node_id)
            
            # Update node health
            node_health.last_check_time = current_time
            node_health.latency_us = health_result["latency_us"]
            node_health.throughput_ops = health_result["throughput_ops"]
            node_health.error_rate = health_result["error_rate"]
            node_health.cpu_utilization = health_result["cpu_utilization"]
            node_health.memory_utilization = health_result["memory_utilization"]
            node_health.network_utilization = health_result["network_utilization"]
            node_health.active_connections = health_result["active_connections"]
            node_health.data_lag_seconds = health_result["data_lag_seconds"]
            
            # Evaluate health status
            is_healthy = await self._evaluate_node_health(node_id, health_result)
            
            if is_healthy:
                if node_health.status == NodeStatus.FAILED:
                    # Node has recovered
                    node_health.status = NodeStatus.ACTIVE
                    node_health.consecutive_failures = 0
                    self.logger.info(f"Node {node_id} has recovered")
                
            else:
                # Node is unhealthy
                node_health.consecutive_failures += 1
                node_health.last_failure_time = current_time
                node_health.failure_reasons.append(health_result.get("failure_reason", "unknown"))
                
                # Check if failover threshold is met
                config = self._get_config_for_node(node_id)
                if config and node_health.consecutive_failures >= config.consecutive_failures_threshold:
                    if node_health.status != NodeStatus.FAILED:
                        node_health.status = NodeStatus.FAILED
                        await self._trigger_failover(node_id, FailoverTrigger.NODE_FAILURE)
            
            # Update cache
            self.health_check_cache[node_id] = (current_time, is_healthy)
            
        except Exception as e:
            self.logger.error(f"Error checking health for node {node_id}: {e}")
            await self._handle_health_check_failure(node_id, str(e))
    
    async def _perform_health_check(self, node_id: str) -> Dict[str, Any]:
        """Perform actual health check measurements"""
        
        # Simulate realistic health check with some variance
        import random
        
        # Base values with realistic variance
        base_latency = 200 + random.uniform(-50, 100)  # 150-300μs
        base_throughput = 10000 + random.uniform(-2000, 5000)  # 8k-15k ops/sec
        
        # Simulate occasional performance issues
        if random.random() < 0.05:  # 5% chance of performance issue
            base_latency *= 3  # High latency
            base_throughput *= 0.3  # Low throughput
            failure_reason = "performance_degradation"
        else:
            failure_reason = None
        
        # Simulate node failures
        if random.random() < 0.01:  # 1% chance of node failure
            failure_reason = "node_unresponsive"
            base_latency = 999999  # Timeout
            base_throughput = 0
        
        return {
            "latency_us": base_latency,
            "throughput_ops": base_throughput,
            "error_rate": max(0, random.uniform(-0.001, 0.01)),
            "cpu_utilization": random.uniform(30, 80),
            "memory_utilization": random.uniform(40, 75),
            "network_utilization": random.uniform(20, 60),
            "active_connections": random.randint(100, 1000),
            "data_lag_seconds": random.uniform(0, 2),
            "failure_reason": failure_reason
        }
    
    async def _evaluate_node_health(self, node_id: str, health_result: Dict[str, Any]) -> bool:
        """Evaluate if node is healthy based on health check results"""
        
        config = self._get_config_for_node(node_id)
        if not config:
            return True  # No config, assume healthy
        
        # Check each threshold
        if health_result["latency_us"] > config.max_latency_us:
            return False
        
        if health_result["throughput_ops"] < config.min_throughput_ops:
            return False
        
        if health_result["error_rate"] > config.max_error_rate:
            return False
        
        if health_result["cpu_utilization"] > config.max_cpu_utilization:
            return False
        
        if health_result.get("failure_reason"):
            return False
        
        return True
    
    def _get_config_for_node(self, node_id: str) -> Optional[FailoverConfiguration]:
        """Get failover configuration for specific node"""
        
        # In a real implementation, nodes would be mapped to deployments
        # For simulation, use first available config
        return next(iter(self.failover_configs.values())) if self.failover_configs else None
    
    async def _handle_health_check_failure(self, node_id: str, reason: str):
        """Handle health check failure"""
        
        if node_id in self.node_health:
            node_health = self.node_health[node_id]
            node_health.consecutive_failures += 1
            node_health.last_failure_time = time.time()
            node_health.failure_reasons.append(reason)
    
    async def _evaluate_failover_conditions(self):
        """Evaluate if failover conditions are met"""
        
        for deployment_name, config in self.failover_configs.items():
            deployment_nodes = [
                node_id for node_id, health in self.node_health.items()
                # In real implementation, filter by deployment
            ]
            
            failed_nodes = [
                node_id for node_id in deployment_nodes
                if self.node_health[node_id].status == NodeStatus.FAILED
            ]
            
            active_nodes = [
                node_id for node_id in deployment_nodes
                if self.node_health[node_id].status == NodeStatus.ACTIVE
            ]
            
            # Check if we have enough active nodes
            if len(active_nodes) < config.required_replica_count:
                if failed_nodes:
                    await self._trigger_deployment_failover(deployment_name, failed_nodes)
    
    async def _trigger_failover(
        self, 
        node_id: str, 
        trigger: FailoverTrigger,
        strategy: FailoverStrategy = FailoverStrategy.GRACEFUL
    ):
        """Trigger failover for specific node"""
        
        event_id = f"failover_{node_id}_{int(time.time())}"
        
        # Check if failover is already in progress for this node
        active_failover = next(
            (event for event in self.active_failovers.values() if node_id in event.failed_nodes),
            None
        )
        
        if active_failover:
            self.logger.warning(f"Failover already in progress for node {node_id}")
            return
        
        self.logger.warning(f"Triggering failover for node {node_id}: {trigger.value}")
        
        # Find target nodes for failover
        target_nodes = await self._select_failover_targets(node_id)
        
        if not target_nodes:
            self.logger.error(f"No suitable failover targets found for node {node_id}")
            return
        
        # Create failover event
        failover_event = FailoverEvent(
            event_id=event_id,
            trigger=trigger,
            strategy=strategy,
            start_time=time.time(),
            failed_nodes=[node_id],
            target_nodes=target_nodes,
            affected_connections=self.node_health[node_id].active_connections
        )
        
        self.active_failovers[event_id] = failover_event
        
        # Execute failover asynchronously
        asyncio.create_task(self._execute_failover(failover_event))
    
    async def _select_failover_targets(self, failed_node_id: str) -> List[str]:
        """Select target nodes for failover"""
        
        # Find healthy nodes that can take over
        candidate_nodes = []
        
        for node_id, health in self.node_health.items():
            if (node_id != failed_node_id and 
                health.status in [NodeStatus.ACTIVE, NodeStatus.STANDBY] and
                health.cpu_utilization < 70 and  # Has capacity
                health.error_rate < 0.005):      # Low error rate
                
                candidate_nodes.append(node_id)
        
        # Sort by health score (lower latency, higher throughput is better)
        candidate_nodes.sort(key=lambda nid: (
            self.node_health[nid].latency_us,
            -self.node_health[nid].throughput_ops
        ))
        
        # Return best candidates (up to 2 for redundancy)
        return candidate_nodes[:2]
    
    async def _execute_failover(self, failover_event: FailoverEvent):
        """Execute failover process"""
        
        event_id = failover_event.event_id
        
        try:
            self.logger.info(f"Executing failover: {event_id}")
            
            # Step 1: Data consistency check
            consistency_ok = await self._ensure_data_consistency(failover_event)
            if not consistency_ok:
                failover_event.steps_failed.append("data_consistency_check")
                failover_event.data_loss_risk = "moderate"
            else:
                failover_event.steps_completed.append("data_consistency_check")
            
            # Step 2: Prepare target nodes
            preparation_ok = await self._prepare_failover_targets(failover_event)
            if not preparation_ok:
                failover_event.steps_failed.append("target_preparation")
                raise Exception("Failed to prepare failover targets")
            
            failover_event.steps_completed.append("target_preparation")
            
            # Step 3: Drain connections (if graceful)
            if failover_event.strategy == FailoverStrategy.GRACEFUL:
                drain_ok = await self._drain_connections(failover_event)
                if not drain_ok:
                    failover_event.steps_failed.append("connection_draining")
                    # Continue with immediate failover
                    failover_event.strategy = FailoverStrategy.IMMEDIATE
                else:
                    failover_event.steps_completed.append("connection_draining")
            
            # Step 4: Switch traffic
            switch_ok = await self._switch_traffic(failover_event)
            if not switch_ok:
                failover_event.steps_failed.append("traffic_switch")
                raise Exception("Failed to switch traffic")
            
            failover_event.steps_completed.append("traffic_switch")
            
            # Step 5: Update node statuses
            await self._update_node_statuses(failover_event)
            failover_event.steps_completed.append("status_update")
            
            # Step 6: Send notifications
            await self._send_failover_notifications(failover_event)
            failover_event.steps_completed.append("notifications")
            
            # Complete failover
            failover_event.end_time = time.time()
            failover_event.duration_seconds = failover_event.end_time - failover_event.start_time
            failover_event.recovery_status = "completed"
            
            self.logger.info(f"Failover completed: {event_id} in {failover_event.duration_seconds:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failover failed: {event_id}: {e}")
            failover_event.recovery_status = "failed"
            failover_event.steps_failed.append(f"execution_error: {str(e)}")
            
            # Attempt rollback if possible
            if failover_event.steps_completed:
                failover_event.rollback_required = True
                await self._attempt_rollback(failover_event)
        
        finally:
            # Move to history and cleanup
            self.failover_events.append(failover_event)
            if event_id in self.active_failovers:
                del self.active_failovers[event_id]
    
    async def _ensure_data_consistency(self, failover_event: FailoverEvent) -> bool:
        """Ensure data consistency before failover"""
        
        try:
            # Check data synchronization status
            failed_node = failover_event.failed_nodes[0]
            
            # Simulate consistency check
            await asyncio.sleep(0.1)  # Simulated check time
            
            # For simulation, assume consistency is ok most of the time
            import random
            return random.random() > 0.1  # 90% success rate
            
        except Exception as e:
            self.logger.error(f"Data consistency check failed: {e}")
            return False
    
    async def _prepare_failover_targets(self, failover_event: FailoverEvent) -> bool:
        """Prepare target nodes for failover"""
        
        try:
            for target_node in failover_event.target_nodes:
                # Update node status to active
                if target_node in self.node_health:
                    self.node_health[target_node].status = NodeStatus.ACTIVE
                
                # Simulate preparation time
                await asyncio.sleep(0.05)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Target preparation failed: {e}")
            return False
    
    async def _drain_connections(self, failover_event: FailoverEvent) -> bool:
        """Drain connections from failed nodes"""
        
        try:
            failed_node = failover_event.failed_nodes[0]
            node_health = self.node_health[failed_node]
            
            # Update status to draining
            node_health.status = NodeStatus.DRAINING
            
            # Simulate connection draining
            drain_time = min(10.0, node_health.active_connections * 0.01)  # Scale with connections
            await asyncio.sleep(drain_time)
            
            # Update connection count
            node_health.active_connections = 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection draining failed: {e}")
            return False
    
    async def _switch_traffic(self, failover_event: FailoverEvent) -> bool:
        """Switch traffic to target nodes"""
        
        try:
            # Simulate traffic switching
            await asyncio.sleep(0.2)  # Simulated switch time
            
            # Calculate downtime
            if failover_event.strategy == FailoverStrategy.IMMEDIATE:
                failover_event.downtime_seconds = 0.5
            elif failover_event.strategy == FailoverStrategy.GRACEFUL:
                failover_event.downtime_seconds = 0.1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Traffic switching failed: {e}")
            return False
    
    async def _update_node_statuses(self, failover_event: FailoverEvent):
        """Update node statuses after failover"""
        
        # Mark failed nodes as failed
        for node_id in failover_event.failed_nodes:
            if node_id in self.node_health:
                self.node_health[node_id].status = NodeStatus.FAILED
        
        # Mark target nodes as active
        for node_id in failover_event.target_nodes:
            if node_id in self.node_health:
                self.node_health[node_id].status = NodeStatus.ACTIVE
    
    async def _send_failover_notifications(self, failover_event: FailoverEvent):
        """Send failover notifications"""
        
        # In production, send actual notifications
        self.logger.info(f"Failover notification: {failover_event.event_id}")
        self.logger.info(f"Failed nodes: {failover_event.failed_nodes}")
        self.logger.info(f"Target nodes: {failover_event.target_nodes}")
        self.logger.info(f"Downtime: {failover_event.downtime_seconds:.2f}s")
    
    async def _attempt_rollback(self, failover_event: FailoverEvent):
        """Attempt to rollback failed failover"""
        
        try:
            self.logger.warning(f"Attempting rollback for failover: {failover_event.event_id}")
            
            # Reverse the steps that were completed
            if "traffic_switch" in failover_event.steps_completed:
                # Switch traffic back
                await asyncio.sleep(0.1)
            
            if "target_preparation" in failover_event.steps_completed:
                # Reset target nodes to standby
                for node_id in failover_event.target_nodes:
                    if node_id in self.node_health:
                        self.node_health[node_id].status = NodeStatus.STANDBY
            
            self.logger.info(f"Rollback completed for failover: {failover_event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
    
    async def _trigger_deployment_failover(self, deployment_name: str, failed_nodes: List[str]):
        """Trigger failover for entire deployment"""
        
        event_id = f"deployment_failover_{deployment_name}_{int(time.time())}"
        
        self.logger.warning(f"Triggering deployment failover: {deployment_name}")
        
        # Find all available standby nodes
        standby_nodes = [
            node_id for node_id, health in self.node_health.items()
            if health.status == NodeStatus.STANDBY
        ]
        
        if len(standby_nodes) < 1:
            self.logger.error(f"No standby nodes available for deployment failover: {deployment_name}")
            return
        
        failover_event = FailoverEvent(
            event_id=event_id,
            trigger=FailoverTrigger.RESOURCE_EXHAUSTION,
            strategy=FailoverStrategy.IMMEDIATE,
            start_time=time.time(),
            failed_nodes=failed_nodes,
            target_nodes=standby_nodes[:2],  # Use up to 2 standby nodes
            affected_connections=sum(self.node_health[nid].active_connections for nid in failed_nodes)
        )
        
        self.active_failovers[event_id] = failover_event
        asyncio.create_task(self._execute_failover(failover_event))
    
    async def _consistency_monitoring_loop(self):
        """Monitor data consistency across edge nodes"""
        
        while self.monitoring_active:
            try:
                for deployment_name, state in self.consistency_state.items():
                    await self._check_data_consistency(deployment_name)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Consistency monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_data_consistency(self, deployment_name: str):
        """Check data consistency for deployment"""
        
        try:
            state = self.consistency_state[deployment_name]
            
            # Simulate consistency checking
            # In production, this would check actual data versions
            current_time = time.time()
            
            # Update version vectors (simulation)
            for node_id in state.node_versions.keys():
                if node_id in self.node_health and self.node_health[node_id].status == NodeStatus.ACTIVE:
                    # Simulate version progression
                    state.node_versions[node_id] += 1
            
            # Check for version lag
            if state.node_versions:
                max_version = max(state.node_versions.values())
                min_version = min(state.node_versions.values())
                state.max_version_lag = max_version - min_version
                
                # Identify nodes that are out of sync
                state.nodes_out_of_sync = [
                    node_id for node_id, version in state.node_versions.items()
                    if max_version - version > 5  # More than 5 versions behind
                ]
            
            # Check if critical data is inconsistent
            config = self.failover_configs.get(deployment_name)
            if config and state.max_version_lag > 10:  # Significant lag
                state.critical_data_inconsistent = True
                self.logger.warning(f"Critical data inconsistency detected in {deployment_name}")
            
            state.timestamp = current_time
            
        except Exception as e:
            self.logger.error(f"Error checking consistency for {deployment_name}: {e}")
    
    async def _recovery_monitoring_loop(self):
        """Monitor recovery of failed nodes"""
        
        while self.monitoring_active:
            try:
                failed_nodes = [
                    node_id for node_id, health in self.node_health.items()
                    if health.status == NodeStatus.FAILED
                ]
                
                for node_id in failed_nodes:
                    await self._check_node_recovery(node_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Recovery monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_node_recovery(self, node_id: str):
        """Check if failed node has recovered"""
        
        try:
            # Perform health check on failed node
            health_result = await self._perform_health_check(node_id)
            
            # Check if node is now healthy
            is_healthy = await self._evaluate_node_health(node_id, health_result)
            
            if is_healthy:
                # Node has recovered, update status
                node_health = self.node_health[node_id]
                node_health.status = NodeStatus.STANDBY  # Start as standby
                node_health.consecutive_failures = 0
                node_health.failure_reasons.clear()
                
                self.logger.info(f"Node {node_id} has recovered and is now on standby")
        
        except Exception as e:
            self.logger.debug(f"Recovery check failed for node {node_id}: {e}")
    
    async def _cleanup_monitoring_loop(self):
        """Cleanup old events and cache entries"""
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Clean up old events (keep last 100)
                if len(self.failover_events) > 100:
                    self.failover_events = self.failover_events[-100:]
                
                # Clean up old health check cache entries
                cutoff_time = current_time - 60  # 1 minute
                self.health_check_cache = {
                    node_id: (timestamp, healthy)
                    for node_id, (timestamp, healthy) in self.health_check_cache.items()
                    if timestamp > cutoff_time
                }
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def manual_failover(
        self,
        node_id: str,
        strategy: FailoverStrategy = FailoverStrategy.GRACEFUL,
        reason: str = "Manual operator intervention"
    ) -> str:
        """Manually trigger failover for specific node"""
        
        if node_id not in self.node_health:
            raise ValueError(f"Node {node_id} not found")
        
        # Check if failover is already in progress
        active_failover = next(
            (event for event in self.active_failovers.values() if node_id in event.failed_nodes),
            None
        )
        
        if active_failover:
            raise ValueError(f"Failover already in progress for node {node_id}")
        
        self.logger.info(f"Manual failover requested for node {node_id}: {reason}")
        
        # Mark node for failover
        self.node_health[node_id].status = NodeStatus.MAINTENANCE
        
        # Trigger failover
        await self._trigger_failover(node_id, FailoverTrigger.MANUAL_TRIGGER, strategy)
        
        return f"Manual failover initiated for node {node_id}"
    
    async def get_failover_status(self) -> Dict[str, Any]:
        """Get comprehensive failover status"""
        
        active_nodes = [nid for nid, h in self.node_health.items() if h.status == NodeStatus.ACTIVE]
        failed_nodes = [nid for nid, h in self.node_health.items() if h.status == NodeStatus.FAILED]
        standby_nodes = [nid for nid, h in self.node_health.items() if h.status == NodeStatus.STANDBY]
        
        return {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "active_failovers": len(self.active_failovers),
            "node_summary": {
                "total_nodes": len(self.node_health),
                "active_nodes": len(active_nodes),
                "failed_nodes": len(failed_nodes),
                "standby_nodes": len(standby_nodes),
                "maintenance_nodes": len([nid for nid, h in self.node_health.items() if h.status == NodeStatus.MAINTENANCE])
            },
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "trigger": event.trigger.value,
                    "duration_seconds": event.duration_seconds,
                    "failed_nodes": event.failed_nodes,
                    "target_nodes": event.target_nodes,
                    "recovery_status": event.recovery_status
                }
                for event in self.failover_events[-10:]  # Last 10 events
            ],
            "consistency_status": {
                deployment: {
                    "max_version_lag": state.max_version_lag,
                    "nodes_out_of_sync": len(state.nodes_out_of_sync),
                    "critical_inconsistency": state.critical_data_inconsistent
                }
                for deployment, state in self.consistency_state.items()
            },
            "node_health": {
                node_id: {
                    "status": health.status.value,
                    "latency_us": health.latency_us,
                    "throughput_ops": health.throughput_ops,
                    "error_rate": health.error_rate,
                    "consecutive_failures": health.consecutive_failures,
                    "last_check": health.last_check_time
                }
                for node_id, health in self.node_health.items()
            }
        }
    
    def stop_monitoring(self):
        """Stop failover monitoring"""
        
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        self.monitoring_tasks.clear()
        self.logger.info("Edge failover monitoring stopped")
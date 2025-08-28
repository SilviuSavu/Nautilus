#!/usr/bin/env python3
"""
Nested Negative Feedback Loop Controller for Nautilus Trading Platform

Implements hierarchical PID control algorithms to create self-optimizing,
self-stabilizing engine coordination patterns across the triple messagebus
architecture with intelligent cache management.

Hierarchical Structure:
- Level 1 (Inner Loops): Engine-local caching and performance optimization
- Level 2 (Middle Loops): MessageBus traffic optimization across engine groups  
- Level 3 (Outer Loops): System-wide orchestration and resource allocation

Author: BMad Orchestrator
Performance Target: <0.18ms average latency (10x improvement from 1.8ms)
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
import json
import statistics
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackLoopLevel(Enum):
    """Hierarchical feedback loop levels"""
    INNER = "inner"      # Engine-local optimization (sub-ms)
    MIDDLE = "middle"    # Bus-level coordination (10-100ms)
    OUTER = "outer"      # System-wide orchestration (1-10s)


class ControllerState(Enum):
    """Controller operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class FeedbackSignalType(Enum):
    """Types of feedback signals for control loops"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"  
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_PRESSURE = "memory_pressure"
    ERROR_RATE = "error_rate"
    ACCURACY = "accuracy"
    RISK_SCORE = "risk_score"
    PL_IMPACT = "pl_impact"


@dataclass
class FeedbackSignal:
    """Individual feedback signal with metadata"""
    signal_type: FeedbackSignalType
    value: float
    timestamp: float
    source_engine: str
    loop_level: FeedbackLoopLevel
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIDParameters:
    """PID controller parameters with auto-tuning capability"""
    kp: float = 1.0      # Proportional gain
    ki: float = 0.1      # Integral gain  
    kd: float = 0.01     # Derivative gain
    setpoint: float = 1.0
    min_output: float = -1.0
    max_output: float = 1.0
    windup_guard: float = 10.0
    
    # Auto-tuning parameters
    auto_tune: bool = True
    tune_period: float = 60.0  # Auto-tune every 60 seconds
    stability_threshold: float = 0.05


class PIDController:
    """Advanced PID controller with auto-tuning and anti-windup"""
    
    def __init__(self, params: PIDParameters):
        self.params = params
        self.reset()
        self.last_tune_time = 0
        self.performance_history = deque(maxlen=1000)
        
    def reset(self):
        """Reset controller state"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, current_value: float, current_time: float) -> float:
        """Calculate PID output with anti-windup and auto-tuning"""
        
        # Calculate error
        error = self.params.setpoint - current_value
        
        # Calculate time delta
        if self.last_time is None:
            dt = 0.001  # 1ms default
        else:
            dt = current_time - self.last_time
            dt = max(dt, 0.0001)  # Prevent division by zero
            
        self.last_time = current_time
        
        # Proportional term
        proportional = self.params.kp * error
        
        # Integral term with windup guard
        self.integral += error * dt
        if self.integral > self.params.windup_guard:
            self.integral = self.params.windup_guard
        elif self.integral < -self.params.windup_guard:
            self.integral = -self.params.windup_guard
            
        integral_term = self.params.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        derivative_term = self.params.kd * derivative
        
        # Calculate total output
        output = proportional + integral_term + derivative_term
        
        # Clamp output to bounds
        output = max(min(output, self.params.max_output), self.params.min_output)
        
        # Store for next iteration
        self.previous_error = error
        
        # Store performance metrics for auto-tuning
        self.performance_history.append({
            'timestamp': current_time,
            'error': abs(error),
            'output': output,
            'setpoint': self.params.setpoint,
            'actual': current_value
        })
        
        # Auto-tune if enabled and enough time has passed
        if (self.params.auto_tune and 
            current_time - self.last_tune_time > self.params.tune_period and
            len(self.performance_history) > 100):
            self._auto_tune()
            self.last_tune_time = current_time
            
        return output
        
    def _auto_tune(self):
        """Ziegler-Nichols inspired auto-tuning algorithm"""
        try:
            # Calculate recent performance metrics
            recent_errors = [h['error'] for h in list(self.performance_history)[-100:]]
            error_std = statistics.stdev(recent_errors)
            error_mean = statistics.mean(recent_errors)
            
            # Check if system is stable
            if error_std < self.params.stability_threshold:
                # System is stable, slightly increase responsiveness
                self.params.kp *= 1.02
                self.params.kd *= 1.01
                logger.debug("Auto-tune: System stable, increased responsiveness")
            elif error_std > error_mean:
                # System is oscillating, reduce gains
                self.params.kp *= 0.95
                self.params.kd *= 0.9
                self.params.ki *= 0.98
                logger.debug("Auto-tune: System oscillating, reduced gains")
            else:
                # System needs more aggressive control
                self.params.ki *= 1.05
                logger.debug("Auto-tune: Increased integral gain")
                
        except (statistics.StatisticsError, ValueError) as e:
            logger.debug(f"Auto-tune skipped: {e}")


@dataclass
class LoopConfiguration:
    """Configuration for a specific feedback loop"""
    loop_id: str
    level: FeedbackLoopLevel
    signal_types: List[FeedbackSignalType]
    target_engines: List[str]
    pid_params: PIDParameters
    update_frequency: float  # Hz
    enabled: bool = True
    
    # Performance thresholds
    emergency_threshold: float = 10.0  # 10x normal value triggers emergency
    degraded_threshold: float = 3.0    # 3x normal value triggers degraded mode


class FeedbackLoopController:
    """
    Hierarchical Nested Negative Feedback Loop Controller
    
    Manages three levels of feedback loops:
    1. Inner Loops (Engine-Local): <1ms response time
    2. Middle Loops (Bus-Level): 10-100ms response time  
    3. Outer Loops (System-Wide): 1-10s response time
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.state = ControllerState.INITIALIZING
        self.loops: Dict[str, LoopConfiguration] = {}
        self.controllers: Dict[str, PIDController] = {}
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.control_actions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance metrics
        self.performance_metrics = {
            'loops_processed': 0,
            'control_actions_taken': 0,
            'emergency_interventions': 0,
            'average_loop_time': 0.0,
            'system_stability_score': 1.0
        }
        
        # Threading for different loop levels
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Callback registry for control actions
        self.action_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Load default configuration
        self._load_default_configuration()
        
        logger.info("🎛️ FeedbackLoopController initialized with nested loop architecture")
        
    def _load_default_configuration(self):
        """Load default feedback loop configurations for Nautilus engines"""
        
        # Inner Loop: Analytics Engine Cache Optimization (Port 8100)
        self.add_loop(LoopConfiguration(
            loop_id="analytics_cache_inner",
            level=FeedbackLoopLevel.INNER,
            signal_types=[FeedbackSignalType.CACHE_HIT_RATE, FeedbackSignalType.LATENCY],
            target_engines=["analytics"],
            pid_params=PIDParameters(kp=2.0, ki=0.5, kd=0.1, setpoint=0.95),  # 95% cache hit target
            update_frequency=1000.0  # 1000 Hz for sub-ms response
        ))
        
        # Inner Loop: Risk Engine Response Time (Port 8200)
        self.add_loop(LoopConfiguration(
            loop_id="risk_latency_inner", 
            level=FeedbackLoopLevel.INNER,
            signal_types=[FeedbackSignalType.LATENCY, FeedbackSignalType.QUEUE_DEPTH],
            target_engines=["risk"],
            pid_params=PIDParameters(kp=3.0, ki=0.8, kd=0.2, setpoint=0.0005),  # 0.5ms target
            update_frequency=2000.0  # 2000 Hz for ultra-low latency
        ))
        
        # Inner Loop: ML Engine Prediction Accuracy (Port 8400)
        self.add_loop(LoopConfiguration(
            loop_id="ml_accuracy_inner",
            level=FeedbackLoopLevel.INNER, 
            signal_types=[FeedbackSignalType.ACCURACY, FeedbackSignalType.CPU_UTILIZATION],
            target_engines=["ml"],
            pid_params=PIDParameters(kp=1.5, ki=0.3, kd=0.05, setpoint=0.92),  # 92% accuracy target
            update_frequency=500.0  # 500 Hz for ML optimization
        ))
        
        # Middle Loop: MarketData Bus Traffic Control (Port 6380)
        self.add_loop(LoopConfiguration(
            loop_id="marketdata_bus_middle",
            level=FeedbackLoopLevel.MIDDLE,
            signal_types=[FeedbackSignalType.THROUGHPUT, FeedbackSignalType.QUEUE_DEPTH],
            target_engines=["marketdata", "analytics", "risk", "strategy"],
            pid_params=PIDParameters(kp=1.2, ki=0.2, kd=0.03, setpoint=15000),  # 15k msg/sec target
            update_frequency=50.0  # 50 Hz for bus coordination
        ))
        
        # Middle Loop: Engine Logic Bus Optimization (Port 6381)  
        self.add_loop(LoopConfiguration(
            loop_id="engine_logic_bus_middle",
            level=FeedbackLoopLevel.MIDDLE,
            signal_types=[FeedbackSignalType.LATENCY, FeedbackSignalType.ERROR_RATE],
            target_engines=["risk", "ml", "strategy", "portfolio", "factor"],
            pid_params=PIDParameters(kp=2.5, ki=0.4, kd=0.08, setpoint=0.0008),  # 0.8ms target
            update_frequency=100.0  # 100 Hz for engine coordination
        ))
        
        # Outer Loop: System-Wide Risk Management
        self.add_loop(LoopConfiguration(
            loop_id="system_risk_outer",
            level=FeedbackLoopLevel.OUTER,
            signal_types=[FeedbackSignalType.RISK_SCORE, FeedbackSignalType.PL_IMPACT],
            target_engines=["risk", "collateral", "portfolio", "strategy"],
            pid_params=PIDParameters(kp=0.8, ki=0.1, kd=0.02, setpoint=0.3),  # 30% max risk score
            update_frequency=1.0  # 1 Hz for risk orchestration
        ))
        
        # Outer Loop: Global Resource Allocation
        self.add_loop(LoopConfiguration(
            loop_id="resource_allocation_outer",
            level=FeedbackLoopLevel.OUTER,
            signal_types=[FeedbackSignalType.CPU_UTILIZATION, FeedbackSignalType.MEMORY_PRESSURE],
            target_engines=["all"],  # All 18 engines
            pid_params=PIDParameters(kp=0.5, ki=0.05, kd=0.01, setpoint=0.75),  # 75% max utilization
            update_frequency=0.1  # 0.1 Hz (10s) for global coordination
        ))
        
        logger.info("📋 Loaded 7 default feedback loop configurations")
        
    def add_loop(self, config: LoopConfiguration):
        """Add a new feedback loop configuration"""
        self.loops[config.loop_id] = config
        self.controllers[config.loop_id] = PIDController(config.pid_params)
        logger.debug(f"➕ Added feedback loop: {config.loop_id} ({config.level.value})")
        
    def remove_loop(self, loop_id: str):
        """Remove a feedback loop"""
        if loop_id in self.loops:
            del self.loops[loop_id]
            del self.controllers[loop_id]
            logger.debug(f"➖ Removed feedback loop: {loop_id}")
        
    def register_action_callback(self, loop_id: str, callback: Callable[[str, float], None]):
        """Register callback for control actions"""
        self.action_callbacks[loop_id].append(callback)
        logger.debug(f"🔗 Registered action callback for loop: {loop_id}")
        
    async def start(self):
        """Start all feedback loop processing"""
        if self._running:
            logger.warning("⚠️ FeedbackLoopController already running")
            return
            
        self._running = True
        self.state = ControllerState.ACTIVE
        
        logger.info("🚀 Starting Nested Negative Feedback Loop Controller")
        
        # Start processing tasks for each loop level
        for level in FeedbackLoopLevel:
            task = asyncio.create_task(self._process_loop_level(level))
            self._tasks.append(task)
            
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_system_health())
        self._tasks.append(monitor_task)
        
        logger.info(f"✅ Started {len(self._tasks)} feedback loop processing tasks")
        
    async def stop(self):
        """Stop all feedback loop processing"""
        if not self._running:
            return
            
        self._running = False
        self.state = ControllerState.SHUTDOWN
        
        logger.info("🛑 Stopping Nested Negative Feedback Loop Controller")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        self._tasks.clear()
        logger.info("✅ All feedback loop tasks stopped")
        
    async def process_feedback_signal(self, signal: FeedbackSignal):
        """Process incoming feedback signal and trigger control actions"""
        
        if self.state != ControllerState.ACTIVE:
            return
            
        # Store signal in history
        signal_key = f"{signal.source_engine}_{signal.signal_type.value}"
        self.signal_history[signal_key].append({
            'timestamp': signal.timestamp,
            'value': signal.value,
            'metadata': signal.metadata
        })
        
        # Find matching loops for this signal
        matching_loops = []
        for loop_id, config in self.loops.items():
            if (config.enabled and 
                signal.signal_type in config.signal_types and
                (signal.source_engine in config.target_engines or "all" in config.target_engines)):
                matching_loops.append((loop_id, config))
        
        # Process each matching loop
        for loop_id, config in matching_loops:
            await self._process_loop_signal(loop_id, config, signal)
            
        self.performance_metrics['loops_processed'] += len(matching_loops)
        
    async def _process_loop_signal(self, loop_id: str, config: LoopConfiguration, signal: FeedbackSignal):
        """Process signal for a specific loop"""
        
        controller = self.controllers[loop_id]
        
        # Check for emergency conditions
        if signal.value > config.emergency_threshold * config.pid_params.setpoint:
            logger.critical(f"🚨 Emergency condition in loop {loop_id}: {signal.value}")
            await self._handle_emergency(loop_id, signal)
            return
            
        # Check for degraded conditions
        if signal.value > config.degraded_threshold * config.pid_params.setpoint:
            logger.warning(f"⚠️ Degraded performance in loop {loop_id}: {signal.value}")
            self.state = ControllerState.DEGRADED
            
        # Calculate control action
        control_output = controller.update(signal.value, signal.timestamp)
        
        # Store control action
        self.control_actions[loop_id].append({
            'timestamp': signal.timestamp,
            'input': signal.value,
            'output': control_output,
            'setpoint': config.pid_params.setpoint
        })
        
        # Execute control action
        await self._execute_control_action(loop_id, config, control_output, signal)
        
        self.performance_metrics['control_actions_taken'] += 1
        
    async def _execute_control_action(self, loop_id: str, config: LoopConfiguration, 
                                    control_output: float, signal: FeedbackSignal):
        """Execute control action based on loop level and type"""
        
        action_taken = False
        
        if config.level == FeedbackLoopLevel.INNER:
            # Inner loop actions: Cache tuning, resource allocation
            action_taken = await self._execute_inner_loop_action(loop_id, control_output, signal)
            
        elif config.level == FeedbackLoopLevel.MIDDLE:
            # Middle loop actions: Bus optimization, priority adjustment
            action_taken = await self._execute_middle_loop_action(loop_id, control_output, signal)
            
        elif config.level == FeedbackLoopLevel.OUTER:
            # Outer loop actions: System-wide coordination
            action_taken = await self._execute_outer_loop_action(loop_id, control_output, signal)
            
        # Notify registered callbacks
        for callback in self.action_callbacks[loop_id]:
            try:
                await callback(loop_id, control_output) if asyncio.iscoroutinefunction(callback) else callback(loop_id, control_output)
            except Exception as e:
                logger.error(f"Error in callback for {loop_id}: {e}")
                
        if action_taken:
            logger.debug(f"🎯 Executed control action for {loop_id}: {control_output:.4f}")
            
    async def _execute_inner_loop_action(self, loop_id: str, control_output: float, signal: FeedbackSignal) -> bool:
        """Execute inner loop control actions (engine-local optimization)"""
        
        if "cache" in loop_id:
            # Cache optimization: Adjust TTL, prefetch aggressiveness
            if control_output > 0.1:
                # Increase cache aggressiveness
                cache_action = {
                    'action': 'increase_cache_size',
                    'factor': min(control_output, 2.0),
                    'target_engine': signal.source_engine
                }
            else:
                # Reduce cache overhead
                cache_action = {
                    'action': 'optimize_cache_eviction', 
                    'factor': abs(control_output),
                    'target_engine': signal.source_engine
                }
            logger.debug(f"💾 Cache action: {cache_action}")
            return True
            
        elif "latency" in loop_id:
            # Latency optimization: Adjust threading, batch sizes
            if control_output > 0.1:
                latency_action = {
                    'action': 'increase_parallelism',
                    'factor': min(control_output, 1.5),
                    'target_engine': signal.source_engine
                }
            else:
                latency_action = {
                    'action': 'optimize_batch_size',
                    'factor': abs(control_output),
                    'target_engine': signal.source_engine
                }
            logger.debug(f"⚡ Latency action: {latency_action}")
            return True
            
        return False
        
    async def _execute_middle_loop_action(self, loop_id: str, control_output: float, signal: FeedbackSignal) -> bool:
        """Execute middle loop control actions (bus-level coordination)"""
        
        if "bus" in loop_id:
            # Bus optimization: Adjust message batching, priority
            if control_output > 0.1:
                bus_action = {
                    'action': 'increase_batch_size',
                    'factor': min(control_output, 1.8),
                    'target_bus': 'marketdata' if 'marketdata' in loop_id else 'engine_logic'
                }
            else:
                bus_action = {
                    'action': 'adjust_priority_weights',
                    'factor': abs(control_output),
                    'target_bus': 'marketdata' if 'marketdata' in loop_id else 'engine_logic'
                }
            logger.debug(f"🚌 Bus action: {bus_action}")
            return True
            
        return False
        
    async def _execute_outer_loop_action(self, loop_id: str, control_output: float, signal: FeedbackSignal) -> bool:
        """Execute outer loop control actions (system-wide orchestration)"""
        
        if "risk" in loop_id:
            # Risk management: Adjust position limits, halt trading
            if control_output > 0.5:
                risk_action = {
                    'action': 'reduce_position_limits',
                    'factor': min(control_output, 0.8),
                    'urgency': 'high'
                }
                logger.warning(f"🛡️ Risk action: {risk_action}")
                return True
                
        elif "resource" in loop_id:
            # Resource allocation: Scale engines up/down
            if control_output > 0.3:
                resource_action = {
                    'action': 'scale_engine_resources',
                    'factor': min(control_output, 1.5),
                    'target': 'high_utilization_engines'
                }
                logger.info(f"⚙️ Resource action: {resource_action}")
                return True
                
        return False
        
    async def _handle_emergency(self, loop_id: str, signal: FeedbackSignal):
        """Handle emergency conditions with immediate intervention"""
        
        self.state = ControllerState.EMERGENCY
        self.performance_metrics['emergency_interventions'] += 1
        
        emergency_actions = []
        
        if signal.signal_type == FeedbackSignalType.LATENCY:
            emergency_actions.extend([
                "halt_non_critical_engines",
                "increase_processing_priority",
                "flush_message_queues"
            ])
        elif signal.signal_type == FeedbackSignalType.RISK_SCORE:
            emergency_actions.extend([
                "halt_all_trading",
                "liquidate_high_risk_positions", 
                "notify_risk_management"
            ])
        elif signal.signal_type == FeedbackSignalType.ERROR_RATE:
            emergency_actions.extend([
                "restart_affected_engines",
                "activate_backup_systems",
                "isolate_error_source"
            ])
            
        logger.critical(f"🚨 EMERGENCY PROTOCOL ACTIVATED for {loop_id}")
        logger.critical(f"🚨 Signal: {signal.signal_type.value} = {signal.value}")  
        logger.critical(f"🚨 Actions: {emergency_actions}")
        
        # Execute emergency protocol (would integrate with actual engine controls)
        for action in emergency_actions:
            logger.critical(f"🚨 EXECUTING: {action}")
            await asyncio.sleep(0.001)  # Simulate action execution
            
    async def _process_loop_level(self, level: FeedbackLoopLevel):
        """Process all loops at a specific level"""
        
        level_loops = [(loop_id, config) for loop_id, config in self.loops.items() 
                      if config.level == level and config.enabled]
                      
        if not level_loops:
            return
            
        logger.info(f"🔄 Processing {len(level_loops)} loops at {level.value} level")
        
        while self._running:
            start_time = time.perf_counter()
            
            # Calculate sleep time for this level
            if level == FeedbackLoopLevel.INNER:
                sleep_time = 0.001  # 1ms for inner loops
            elif level == FeedbackLoopLevel.MIDDLE:
                sleep_time = 0.02   # 20ms for middle loops  
            else:
                sleep_time = 1.0    # 1s for outer loops
                
            # Process level-specific maintenance tasks
            await self._maintain_loop_level(level, level_loops)
            
            # Calculate actual processing time
            process_time = time.perf_counter() - start_time
            self.performance_metrics['average_loop_time'] = (
                self.performance_metrics['average_loop_time'] * 0.9 + process_time * 0.1
            )
            
            # Sleep until next iteration
            if process_time < sleep_time:
                await asyncio.sleep(sleep_time - process_time)
                
    async def _maintain_loop_level(self, level: FeedbackLoopLevel, loops: List[Tuple[str, LoopConfiguration]]):
        """Perform maintenance tasks for a specific loop level"""
        
        if level == FeedbackLoopLevel.INNER:
            # Inner loop maintenance: Fast cache optimization
            for loop_id, config in loops:
                if "cache" in loop_id:
                    await self._optimize_cache_inner_loop(loop_id, config)
                    
        elif level == FeedbackLoopLevel.MIDDLE:
            # Middle loop maintenance: Bus health monitoring
            for loop_id, config in loops:
                if "bus" in loop_id:
                    await self._monitor_bus_health(loop_id, config)
                    
        elif level == FeedbackLoopLevel.OUTER:
            # Outer loop maintenance: System-wide analysis
            await self._analyze_system_stability()
            
    async def _optimize_cache_inner_loop(self, loop_id: str, config: LoopConfiguration):
        """Optimize cache performance for inner loops"""
        
        # Analyze recent cache performance
        cache_signals = self.signal_history.get(f"analytics_{FeedbackSignalType.CACHE_HIT_RATE.value}", deque())
        
        if len(cache_signals) > 10:
            recent_hit_rates = [s['value'] for s in list(cache_signals)[-10:]]
            avg_hit_rate = statistics.mean(recent_hit_rates)
            
            # Predictive cache warming based on patterns
            if avg_hit_rate < 0.90:  # Below 90% hit rate
                logger.debug(f"🔄 Cache hit rate low ({avg_hit_rate:.2%}), triggering warming")
                # Would trigger predictive cache warming here
                
    async def _monitor_bus_health(self, loop_id: str, config: LoopConfiguration):
        """Monitor message bus health for middle loops"""
        
        # Monitor queue depths and throughput
        throughput_signals = self.signal_history.get(f"bus_{FeedbackSignalType.THROUGHPUT.value}", deque())
        
        if len(throughput_signals) > 5:
            recent_throughput = [s['value'] for s in list(throughput_signals)[-5:]]
            avg_throughput = statistics.mean(recent_throughput)
            
            # Check if throughput is declining
            if len(recent_throughput) >= 2 and recent_throughput[-1] < recent_throughput[0] * 0.8:
                logger.warning(f"📉 Bus throughput declining: {avg_throughput:.0f} msg/sec")
                
    async def _analyze_system_stability(self):
        """Analyze overall system stability for outer loops"""
        
        # Calculate system-wide stability score
        stability_factors = []
        
        # Factor 1: Control loop convergence
        converged_loops = 0
        total_loops = len(self.loops)
        
        for loop_id, controller in self.controllers.items():
            if len(controller.performance_history) > 10:
                recent_errors = [h['error'] for h in list(controller.performance_history)[-10:]]
                error_trend = statistics.stdev(recent_errors) if len(recent_errors) > 1 else 1.0
                if error_trend < 0.1:  # Low error variance indicates convergence
                    converged_loops += 1
                    
        convergence_factor = converged_loops / max(total_loops, 1)
        stability_factors.append(convergence_factor)
        
        # Factor 2: Emergency intervention frequency
        emergency_factor = max(0, 1.0 - (self.performance_metrics['emergency_interventions'] / 100.0))
        stability_factors.append(emergency_factor)
        
        # Factor 3: Overall system state
        state_factor = {
            ControllerState.ACTIVE: 1.0,
            ControllerState.DEGRADED: 0.7,
            ControllerState.EMERGENCY: 0.3,
            ControllerState.INITIALIZING: 0.5,
            ControllerState.SHUTDOWN: 0.0
        }.get(self.state, 0.5)
        stability_factors.append(state_factor)
        
        # Calculate weighted stability score
        self.performance_metrics['system_stability_score'] = statistics.mean(stability_factors)
        
        logger.debug(f"📊 System stability score: {self.performance_metrics['system_stability_score']:.3f}")
        
    async def _monitor_system_health(self):
        """Monitor overall system health and performance"""
        
        while self._running:
            try:
                # Log performance metrics every 30 seconds
                await asyncio.sleep(30)
                
                metrics = self.performance_metrics.copy()
                logger.info(f"📊 FEEDBACK LOOP CONTROLLER METRICS:")
                logger.info(f"   🔄 Loops Processed: {metrics['loops_processed']}")
                logger.info(f"   🎯 Control Actions: {metrics['control_actions_taken']}")
                logger.info(f"   🚨 Emergency Interventions: {metrics['emergency_interventions']}")
                logger.info(f"   ⏱️ Avg Loop Time: {metrics['average_loop_time']*1000:.2f}ms")
                logger.info(f"   📈 System Stability: {metrics['system_stability_score']:.3f}")
                logger.info(f"   🏥 Controller State: {self.state.value}")
                
                # Auto-recover from degraded state if stability is high
                if (self.state == ControllerState.DEGRADED and 
                    metrics['system_stability_score'] > 0.95):
                    self.state = ControllerState.ACTIVE
                    logger.info("✅ Automatically recovered to ACTIVE state")
                    
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'state': self.state.value,
            'active_loops': len([c for c in self.loops.values() if c.enabled]),
            'total_loops': len(self.loops),
            'signal_history_size': sum(len(h) for h in self.signal_history.values()),
            'control_history_size': sum(len(h) for h in self.control_actions.values())
        }
        
    def get_loop_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feedback loops"""
        status = {}
        
        for loop_id, config in self.loops.items():
            controller = self.controllers[loop_id]
            
            # Calculate recent performance
            recent_performance = {}
            if len(controller.performance_history) > 0:
                recent_data = list(controller.performance_history)[-10:]
                recent_performance = {
                    'avg_error': statistics.mean([h['error'] for h in recent_data]),
                    'avg_output': statistics.mean([h['output'] for h in recent_data]),
                    'stability': statistics.stdev([h['error'] for h in recent_data]) if len(recent_data) > 1 else 0.0
                }
                
            status[loop_id] = {
                'config': config,
                'pid_params': controller.params,
                'recent_performance': recent_performance,
                'signal_count': len(self.signal_history.get(loop_id, [])),
                'action_count': len(self.control_actions.get(loop_id, []))
            }
            
        return status


# Factory function for easy instantiation
def create_feedback_controller(config_path: Optional[str] = None) -> FeedbackLoopController:
    """Create a configured FeedbackLoopController instance"""
    return FeedbackLoopController(config_path)


# Singleton instance for global access
_global_controller: Optional[FeedbackLoopController] = None

async def get_feedback_controller() -> FeedbackLoopController:
    """Get global feedback loop controller instance"""
    global _global_controller
    
    if _global_controller is None:
        _global_controller = create_feedback_controller()
        await _global_controller.start()
        
    return _global_controller


if __name__ == "__main__":
    # Test the feedback loop controller
    async def test_controller():
        print("🧪 Testing Nested Negative Feedback Loop Controller")
        print("=" * 60)
        
        controller = create_feedback_controller()
        
        try:
            await controller.start()
            
            # Simulate some feedback signals
            test_signals = [
                FeedbackSignal(
                    signal_type=FeedbackSignalType.LATENCY,
                    value=0.002,  # 2ms latency
                    timestamp=time.time(),
                    source_engine="risk",
                    loop_level=FeedbackLoopLevel.INNER
                ),
                FeedbackSignal(
                    signal_type=FeedbackSignalType.CACHE_HIT_RATE,
                    value=0.85,  # 85% cache hit rate
                    timestamp=time.time(),
                    source_engine="analytics", 
                    loop_level=FeedbackLoopLevel.INNER
                ),
                FeedbackSignal(
                    signal_type=FeedbackSignalType.THROUGHPUT,
                    value=12000,  # 12k msg/sec
                    timestamp=time.time(),
                    source_engine="marketdata",
                    loop_level=FeedbackLoopLevel.MIDDLE
                )
            ]
            
            # Process test signals
            for signal in test_signals:
                await controller.process_feedback_signal(signal)
                await asyncio.sleep(0.1)
                
            # Let it run for a few seconds
            await asyncio.sleep(5)
            
            # Print results
            metrics = controller.get_performance_metrics()
            print("\n📊 Test Results:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
                
            loop_status = controller.get_loop_status()
            print(f"\n🔄 Active Loops: {len(loop_status)}")
            
        finally:
            await controller.stop()
    
    asyncio.run(test_controller())
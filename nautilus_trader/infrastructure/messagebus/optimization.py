#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Adaptive Performance Optimization for Enhanced MessageBus.

Provides intelligent performance optimization that automatically adapts to:
- System resource availability
- Message load patterns
- Network conditions
- Trading session characteristics
- Latency requirements and SLA targets
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from nautilus_trader.infrastructure.messagebus.config import MessagePriority


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    BALANCED = "balanced"           # Balance throughput and latency
    LATENCY_FIRST = "latency_first"  # Minimize latency at all costs
    THROUGHPUT_FIRST = "throughput_first"  # Maximize throughput
    RESOURCE_EFFICIENT = "resource_efficient"  # Minimize resource usage
    ADAPTIVE = "adaptive"           # Dynamically adapt based on conditions


class SystemLoad(Enum):
    """System load levels."""
    IDLE = "idle"         # < 20% resource usage
    LIGHT = "light"       # 20-40% resource usage
    MODERATE = "moderate" # 40-70% resource usage
    HEAVY = "heavy"       # 70-90% resource usage
    CRITICAL = "critical" # > 90% resource usage


class NetworkCondition(Enum):
    """Network condition classifications."""
    EXCELLENT = "excellent"  # < 1ms latency, < 0.1% loss
    GOOD = "good"           # 1-5ms latency, < 1% loss
    FAIR = "fair"           # 5-20ms latency, < 5% loss
    POOR = "poor"           # 20-100ms latency, < 10% loss
    DEGRADED = "degraded"   # > 100ms latency, > 10% loss


@dataclass
class SystemResourceState:
    """Current system resource state."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    network_latency_ms: float = 0.0
    network_loss_percent: float = 0.0
    disk_io_percent: float = 0.0
    available_memory_mb: float = 0.0
    load_level: SystemLoad = SystemLoad.IDLE
    network_condition: NetworkCondition = NetworkCondition.EXCELLENT


@dataclass
class PerformanceTarget:
    """Performance optimization targets."""
    max_latency_ms: float = 10.0
    min_throughput_msg_per_sec: float = 1000.0
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 70.0
    max_error_rate: float = 0.01
    priority_latency_targets: Dict[MessagePriority, float] = None
    
    def __post_init__(self):
        if self.priority_latency_targets is None:
            self.priority_latency_targets = {
                MessagePriority.CRITICAL: 0.5,  # 0.5ms
                MessagePriority.HIGH: 2.0,      # 2ms
                MessagePriority.NORMAL: 10.0,   # 10ms
                MessagePriority.LOW: 50.0       # 50ms
            }


@dataclass
class OptimizationAction:
    """Performance optimization action."""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: Dict[str, float]  # metric -> improvement %
    resource_cost: Dict[str, float]        # resource -> cost %
    priority: int = 5  # 1=highest, 10=lowest


class AdaptiveOptimizer:
    """
    Adaptive performance optimizer for Enhanced MessageBus.
    
    Continuously monitors system performance and automatically applies
    optimizations to maintain performance targets under varying conditions.
    """
    
    def __init__(self,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
                 performance_targets: Optional[PerformanceTarget] = None,
                 optimization_interval_seconds: int = 30):
        """
        Initialize adaptive optimizer.
        
        Args:
            optimization_strategy: Optimization approach
            performance_targets: Performance targets to optimize for
            optimization_interval_seconds: How often to run optimizations
        """
        self.strategy = optimization_strategy
        self.targets = performance_targets or PerformanceTarget()
        self.optimization_interval = optimization_interval_seconds
        
        # System state tracking
        self._resource_state = SystemResourceState()
        self._resource_history = deque(maxlen=100)
        self._performance_history = deque(maxlen=100)
        
        # Current optimization parameters
        self._current_params = {
            'batch_size': 100,
            'flush_interval_ms': 100,
            'worker_count': 4,
            'buffer_size': 10000,
            'priority_weights': {p: 1.0 for p in MessagePriority},
            'compression_enabled': False,
            'batching_enabled': True,
            'async_processing': True
        }
        
        # Optimization state
        self._applied_optimizations: List[OptimizationAction] = []
        self._optimization_history = deque(maxlen=50)
        self._performance_baselines = {}
        
        # Learning components
        self._optimization_scores = defaultdict(list)
        self._parameter_effectiveness = defaultdict(lambda: defaultdict(list))
        
        # Monitoring
        self._is_optimizing = False
        self._optimizer_task: Optional[asyncio.Task] = None
        self._total_optimizations = 0
        self._successful_optimizations = 0
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def start_optimization(self) -> None:
        """Start adaptive optimization."""
        if not self._is_optimizing:
            self._is_optimizing = True
            self._optimizer_task = asyncio.create_task(self._optimization_loop())
            self._logger.info(f"Adaptive optimization started with {self.strategy.value} strategy")
    
    def stop_optimization(self) -> None:
        """Stop adaptive optimization."""
        self._is_optimizing = False
        if self._optimizer_task and not self._optimizer_task.done():
            self._optimizer_task.cancel()
        self._logger.info("Adaptive optimization stopped")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._is_optimizing:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._run_optimization_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _run_optimization_cycle(self) -> None:
        """Run single optimization cycle."""
        current_time = time.time()
        
        # Update system resource state
        await self._update_resource_state()
        
        # Analyze current performance vs targets
        performance_gaps = self._analyze_performance_gaps()
        
        if not performance_gaps:
            self._logger.debug("All performance targets met, no optimization needed")
            return
        
        # Generate optimization candidates
        optimization_candidates = self._generate_optimizations(performance_gaps)
        
        # Select and apply best optimization
        if optimization_candidates:
            best_optimization = self._select_best_optimization(optimization_candidates)
            await self._apply_optimization(best_optimization)
            
            # Track optimization
            self._optimization_history.append({
                'timestamp': current_time,
                'optimization': best_optimization,
                'performance_gaps': performance_gaps,
                'resource_state': self._resource_state
            })
            
            self._total_optimizations += 1
    
    async def _update_resource_state(self) -> None:
        """Update current system resource state."""
        # In real implementation, would collect actual system metrics
        # For now, simulate based on optimization strategy
        import psutil
        
        try:
            self._resource_state.cpu_percent = psutil.cpu_percent(interval=1)
            self._resource_state.memory_percent = psutil.virtual_memory().percent
            self._resource_state.available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            # Classify load level
            if self._resource_state.cpu_percent < 20:
                self._resource_state.load_level = SystemLoad.IDLE
            elif self._resource_state.cpu_percent < 40:
                self._resource_state.load_level = SystemLoad.LIGHT
            elif self._resource_state.cpu_percent < 70:
                self._resource_state.load_level = SystemLoad.MODERATE
            elif self._resource_state.cpu_percent < 90:
                self._resource_state.load_level = SystemLoad.HEAVY
            else:
                self._resource_state.load_level = SystemLoad.CRITICAL
            
            # Store in history
            self._resource_history.append({
                'timestamp': time.time(),
                'state': self._resource_state
            })
            
        except Exception as e:
            self._logger.warning(f"Could not update resource state: {e}")
    
    def _analyze_performance_gaps(self) -> Dict[str, float]:
        """Analyze current performance vs targets."""
        gaps = {}
        
        # Analyze recent performance data
        if len(self._performance_history) < 5:
            return gaps  # Need more data
        
        recent_performance = list(self._performance_history)[-10:]
        
        # Calculate average performance metrics
        avg_latency = np.mean([p.get('latency_ms', 0) for p in recent_performance])
        avg_throughput = np.mean([p.get('throughput', 0) for p in recent_performance])
        avg_error_rate = np.mean([p.get('error_rate', 0) for p in recent_performance])
        
        # Check against targets
        if avg_latency > self.targets.max_latency_ms:
            gaps['latency'] = (avg_latency - self.targets.max_latency_ms) / self.targets.max_latency_ms
        
        if avg_throughput < self.targets.min_throughput_msg_per_sec:
            gaps['throughput'] = (self.targets.min_throughput_msg_per_sec - avg_throughput) / self.targets.min_throughput_msg_per_sec
        
        if avg_error_rate > self.targets.max_error_rate:
            gaps['error_rate'] = (avg_error_rate - self.targets.max_error_rate) / self.targets.max_error_rate
        
        # Check resource utilization
        if self._resource_state.cpu_percent > self.targets.max_cpu_percent:
            gaps['cpu_usage'] = (self._resource_state.cpu_percent - self.targets.max_cpu_percent) / self.targets.max_cpu_percent
        
        if self._resource_state.memory_percent > self.targets.max_memory_percent:
            gaps['memory_usage'] = (self._resource_state.memory_percent - self.targets.max_memory_percent) / self.targets.max_memory_percent
        
        return gaps
    
    def _generate_optimizations(self, performance_gaps: Dict[str, float]) -> List[OptimizationAction]:
        """Generate optimization candidates based on performance gaps."""
        optimizations = []
        
        # Latency optimization
        if 'latency' in performance_gaps:
            optimizations.extend(self._generate_latency_optimizations(performance_gaps['latency']))
        
        # Throughput optimization
        if 'throughput' in performance_gaps:
            optimizations.extend(self._generate_throughput_optimizations(performance_gaps['throughput']))
        
        # Resource optimization
        if 'cpu_usage' in performance_gaps or 'memory_usage' in performance_gaps:
            optimizations.extend(self._generate_resource_optimizations(performance_gaps))
        
        # Error rate optimization
        if 'error_rate' in performance_gaps:
            optimizations.extend(self._generate_error_optimizations(performance_gaps['error_rate']))
        
        return optimizations
    
    def _generate_latency_optimizations(self, latency_gap: float) -> List[OptimizationAction]:
        """Generate optimizations to reduce latency."""
        optimizations = []
        
        # Reduce flush interval for faster processing
        if self._current_params['flush_interval_ms'] > 10:
            new_interval = max(10, self._current_params['flush_interval_ms'] * 0.5)
            optimizations.append(OptimizationAction(
                action_type="reduce_flush_interval",
                parameters={'flush_interval_ms': new_interval},
                expected_improvement={'latency': 30.0},
                resource_cost={'cpu': 15.0},
                priority=2
            ))
        
        # Increase worker count for parallel processing
        if self._current_params['worker_count'] < 8:
            new_count = min(8, self._current_params['worker_count'] + 2)
            optimizations.append(OptimizationAction(
                action_type="increase_workers",
                parameters={'worker_count': new_count},
                expected_improvement={'latency': 20.0, 'throughput': 25.0},
                resource_cost={'cpu': 25.0, 'memory': 20.0},
                priority=3
            ))
        
        # Enable async processing if not already enabled
        if not self._current_params['async_processing']:
            optimizations.append(OptimizationAction(
                action_type="enable_async",
                parameters={'async_processing': True},
                expected_improvement={'latency': 40.0, 'throughput': 30.0},
                resource_cost={'memory': 10.0},
                priority=1
            ))
        
        # Reduce batch size for faster processing
        if self._current_params['batch_size'] > 10:
            new_size = max(10, self._current_params['batch_size'] // 2)
            optimizations.append(OptimizationAction(
                action_type="reduce_batch_size",
                parameters={'batch_size': new_size},
                expected_improvement={'latency': 25.0},
                resource_cost={'throughput': -10.0},  # Negative = reduction
                priority=4
            ))
        
        return optimizations
    
    def _generate_throughput_optimizations(self, throughput_gap: float) -> List[OptimizationAction]:
        """Generate optimizations to increase throughput."""
        optimizations = []
        
        # Increase batch size for better throughput
        if self._current_params['batch_size'] < 1000:
            new_size = min(1000, self._current_params['batch_size'] * 2)
            optimizations.append(OptimizationAction(
                action_type="increase_batch_size",
                parameters={'batch_size': new_size},
                expected_improvement={'throughput': 35.0},
                resource_cost={'memory': 15.0, 'latency': 20.0},
                priority=2
            ))
        
        # Increase buffer size to handle more messages
        if self._current_params['buffer_size'] < 50000:
            new_size = min(50000, self._current_params['buffer_size'] * 2)
            optimizations.append(OptimizationAction(
                action_type="increase_buffer_size",
                parameters={'buffer_size': new_size},
                expected_improvement={'throughput': 20.0},
                resource_cost={'memory': 30.0},
                priority=3
            ))
        
        # Enable compression to reduce network overhead
        if not self._current_params['compression_enabled']:
            optimizations.append(OptimizationAction(
                action_type="enable_compression",
                parameters={'compression_enabled': True},
                expected_improvement={'throughput': 15.0, 'network_usage': -25.0},
                resource_cost={'cpu': 10.0},
                priority=4
            ))
        
        # Optimize priority weights for better flow
        optimizations.append(OptimizationAction(
            action_type="optimize_priority_weights",
            parameters={
                'priority_weights': self._calculate_optimal_priority_weights()
            },
            expected_improvement={'throughput': 10.0, 'latency': 5.0},
            resource_cost={},
            priority=5
        ))
        
        return optimizations
    
    def _generate_resource_optimizations(self, performance_gaps: Dict[str, float]) -> List[OptimizationAction]:
        """Generate optimizations to reduce resource usage."""
        optimizations = []
        
        cpu_gap = performance_gaps.get('cpu_usage', 0)
        memory_gap = performance_gaps.get('memory_usage', 0)
        
        if cpu_gap > 0.2:  # > 20% over target
            # Reduce worker count to save CPU
            if self._current_params['worker_count'] > 2:
                new_count = max(2, self._current_params['worker_count'] - 1)
                optimizations.append(OptimizationAction(
                    action_type="reduce_workers",
                    parameters={'worker_count': new_count},
                    expected_improvement={'cpu_usage': -15.0},
                    resource_cost={'throughput': -10.0, 'latency': 10.0},
                    priority=2
                ))
            
            # Increase flush interval to reduce CPU overhead
            new_interval = min(1000, self._current_params['flush_interval_ms'] * 1.5)
            optimizations.append(OptimizationAction(
                action_type="increase_flush_interval",
                parameters={'flush_interval_ms': new_interval},
                expected_improvement={'cpu_usage': -10.0},
                resource_cost={'latency': 15.0},
                priority=3
            ))
        
        if memory_gap > 0.2:  # > 20% over target
            # Reduce buffer size to save memory
            if self._current_params['buffer_size'] > 1000:
                new_size = max(1000, self._current_params['buffer_size'] // 2)
                optimizations.append(OptimizationAction(
                    action_type="reduce_buffer_size",
                    parameters={'buffer_size': new_size},
                    expected_improvement={'memory_usage': -25.0},
                    resource_cost={'throughput': -15.0},
                    priority=2
                ))
            
            # Reduce batch size to save memory
            if self._current_params['batch_size'] > 50:
                new_size = max(50, self._current_params['batch_size'] // 2)
                optimizations.append(OptimizationAction(
                    action_type="reduce_batch_size",
                    parameters={'batch_size': new_size},
                    expected_improvement={'memory_usage': -15.0},
                    resource_cost={'throughput': -10.0},
                    priority=3
                ))
        
        return optimizations
    
    def _generate_error_optimizations(self, error_gap: float) -> List[OptimizationAction]:
        """Generate optimizations to reduce error rate."""
        optimizations = []
        
        # Add retry mechanisms
        optimizations.append(OptimizationAction(
            action_type="enable_retries",
            parameters={'retry_enabled': True, 'max_retries': 3},
            expected_improvement={'error_rate': -50.0},
            resource_cost={'latency': 10.0, 'cpu': 5.0},
            priority=1
        ))
        
        # Reduce load to prevent overload errors
        if self._current_params['worker_count'] > 2:
            optimizations.append(OptimizationAction(
                action_type="reduce_load",
                parameters={'worker_count': self._current_params['worker_count'] - 1},
                expected_improvement={'error_rate': -30.0},
                resource_cost={'throughput': -15.0},
                priority=2
            ))
        
        return optimizations
    
    def _calculate_optimal_priority_weights(self) -> Dict[MessagePriority, float]:
        """Calculate optimal priority weights based on current conditions."""
        weights = {}
        
        if self.strategy == OptimizationStrategy.LATENCY_FIRST:
            weights = {
                MessagePriority.CRITICAL: 10.0,
                MessagePriority.HIGH: 5.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.1
            }
        elif self.strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            weights = {
                MessagePriority.CRITICAL: 2.0,
                MessagePriority.HIGH: 1.5,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.8
            }
        else:  # Balanced or adaptive
            # Adjust based on current load
            if self._resource_state.load_level in [SystemLoad.HEAVY, SystemLoad.CRITICAL]:
                weights = {
                    MessagePriority.CRITICAL: 8.0,
                    MessagePriority.HIGH: 4.0,
                    MessagePriority.NORMAL: 1.0,
                    MessagePriority.LOW: 0.2
                }
            else:
                weights = {
                    MessagePriority.CRITICAL: 4.0,
                    MessagePriority.HIGH: 2.0,
                    MessagePriority.NORMAL: 1.0,
                    MessagePriority.LOW: 0.5
                }
        
        return weights
    
    def _select_best_optimization(self, candidates: List[OptimizationAction]) -> OptimizationAction:
        """Select best optimization from candidates."""
        if not candidates:
            return None
        
        # Score each optimization based on strategy
        scored_optimizations = []
        
        for opt in candidates:
            score = self._score_optimization(opt)
            scored_optimizations.append((score, opt))
        
        # Sort by score (higher is better) and priority (lower is better)
        scored_optimizations.sort(key=lambda x: (x[0], -x[1].priority), reverse=True)
        
        return scored_optimizations[0][1]
    
    def _score_optimization(self, optimization: OptimizationAction) -> float:
        """Score optimization based on expected benefits and costs."""
        benefit_score = 0.0
        cost_score = 0.0
        
        # Calculate benefit score
        for metric, improvement in optimization.expected_improvement.items():
            weight = self._get_metric_weight(metric)
            benefit_score += improvement * weight
        
        # Calculate cost score  
        for resource, cost in optimization.resource_cost.items():
            weight = self._get_resource_weight(resource)
            cost_score += abs(cost) * weight
        
        # Apply strategy-specific scoring
        if self.strategy == OptimizationStrategy.LATENCY_FIRST:
            if 'latency' in optimization.expected_improvement:
                benefit_score *= 2.0  # Double weight for latency improvements
        elif self.strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            if 'throughput' in optimization.expected_improvement:
                benefit_score *= 2.0
        elif self.strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            cost_score *= 2.0  # Double weight for resource costs
        
        # Final score (benefit - cost)
        return benefit_score - cost_score
    
    def _get_metric_weight(self, metric: str) -> float:
        """Get weight for performance metric."""
        weights = {
            'latency': 1.0,
            'throughput': 1.0,
            'error_rate': 2.0,  # Errors are very bad
            'cpu_usage': 0.8,
            'memory_usage': 0.8,
            'network_usage': 0.6
        }
        return weights.get(metric, 0.5)
    
    def _get_resource_weight(self, resource: str) -> float:
        """Get weight for resource cost."""
        weights = {
            'cpu': 1.0,
            'memory': 1.0,
            'network': 0.8,
            'disk': 0.6,
            'latency': 1.2,  # Latency cost is expensive
            'throughput': 1.0
        }
        return weights.get(resource, 0.5)
    
    async def _apply_optimization(self, optimization: OptimizationAction) -> bool:
        """Apply optimization action."""
        if not optimization:
            return False
        
        try:
            # Update parameters
            for param, value in optimization.parameters.items():
                if param in self._current_params:
                    old_value = self._current_params[param]
                    self._current_params[param] = value
                    self._logger.info(f"Updated {param}: {old_value} -> {value}")
                else:
                    self._current_params[param] = value
                    self._logger.info(f"Added parameter {param}: {value}")
            
            # Add to applied optimizations
            self._applied_optimizations.append(optimization)
            
            # Track effectiveness for learning
            self._track_optimization_effectiveness(optimization)
            
            self._successful_optimizations += 1
            self._logger.info(f"Applied optimization: {optimization.action_type}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to apply optimization {optimization.action_type}: {e}")
            return False
    
    def _track_optimization_effectiveness(self, optimization: OptimizationAction) -> None:
        """Track optimization effectiveness for learning."""
        # Store baseline performance before optimization
        if len(self._performance_history) >= 5:
            baseline = {
                'latency': np.mean([p.get('latency_ms', 0) for p in list(self._performance_history)[-5:]]),
                'throughput': np.mean([p.get('throughput', 0) for p in list(self._performance_history)[-5:]]),
                'error_rate': np.mean([p.get('error_rate', 0) for p in list(self._performance_history)[-5:]])
            }
            self._performance_baselines[optimization.action_type] = baseline
    
    def record_performance_metrics(self, 
                                  latency_ms: float,
                                  throughput: float,
                                  error_rate: float,
                                  additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record current performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'latency_ms': latency_ms,
            'throughput': throughput,
            'error_rate': error_rate
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self._performance_history.append(metrics)
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        return dict(self._current_params)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        success_rate = (
            self._successful_optimizations / self._total_optimizations
            if self._total_optimizations > 0 else 0.0
        )
        
        return {
            'strategy': self.strategy.value,
            'total_optimizations': self._total_optimizations,
            'successful_optimizations': self._successful_optimizations,
            'success_rate': success_rate,
            'current_load_level': self._resource_state.load_level.value,
            'current_network_condition': self._resource_state.network_condition.value,
            'applied_optimizations_count': len(self._applied_optimizations),
            'is_optimizing': self._is_optimizing,
            'current_parameters': dict(self._current_params)
        }
    
    def reset_optimizations(self) -> None:
        """Reset all optimizations to default values."""
        self._current_params = {
            'batch_size': 100,
            'flush_interval_ms': 100,
            'worker_count': 4,
            'buffer_size': 10000,
            'priority_weights': {p: 1.0 for p in MessagePriority},
            'compression_enabled': False,
            'batching_enabled': True,
            'async_processing': True
        }
        
        self._applied_optimizations.clear()
        self._optimization_history.clear()
        self._performance_baselines.clear()
        
        self._logger.info("Reset all optimizations to defaults")
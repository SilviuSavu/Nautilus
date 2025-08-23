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
Enhanced MessageBus Benchmarking Suite.

Provides comprehensive performance benchmarking, load testing, and performance
regression detection for Enhanced MessageBus implementations:
- Throughput and latency benchmarks
- Scalability testing with varying loads
- Resource utilization profiling
- Performance regression detection
- Comparative analysis vs standard MessageBus
- Real-world scenario simulation
"""

import asyncio
import gc
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from nautilus_trader.infrastructure.messagebus.config import MessagePriority


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    THROUGHPUT = "throughput"                    # Maximum message throughput
    LATENCY = "latency"                         # End-to-end latency
    SCALABILITY = "scalability"                 # Performance under load
    RESOURCE_USAGE = "resource_usage"           # CPU/Memory utilization
    PRIORITY_FAIRNESS = "priority_fairness"     # Priority queue fairness
    PATTERN_MATCHING = "pattern_matching"       # Pattern matching performance
    BATCH_PROCESSING = "batch_processing"       # Batch processing efficiency
    REAL_WORLD_SIMULATION = "real_world_sim"    # Real trading scenario


class LoadProfile(Enum):
    """Load testing profiles."""
    CONSTANT = "constant"           # Constant message rate
    RAMP_UP = "ramp_up"            # Gradual increase
    SPIKE = "spike"                # Sudden load spikes
    BURSTY = "bursty"              # Intermittent bursts
    REALISTIC = "realistic"        # Market-like patterns


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    benchmark_type: BenchmarkType
    duration_seconds: int = 60
    message_count: int = 100000
    concurrent_producers: int = 1
    concurrent_consumers: int = 1
    message_size_bytes: int = 1024
    batch_size: int = 100
    warmup_seconds: int = 10
    cooldown_seconds: int = 5
    priority_distribution: Dict[MessagePriority, float] = field(default_factory=lambda: {
        MessagePriority.CRITICAL: 0.05,
        MessagePriority.HIGH: 0.15,
        MessagePriority.NORMAL: 0.70,
        MessagePriority.LOW: 0.10
    })
    load_profile: LoadProfile = LoadProfile.CONSTANT
    enable_gc_tracking: bool = True
    enable_resource_monitoring: bool = True


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    benchmark_type: BenchmarkType
    config: BenchmarkConfig
    start_time: float
    end_time: float
    duration: float
    
    # Performance metrics
    messages_processed: int = 0
    messages_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    
    # Resource usage
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    gc_collections: int = 0
    
    # Error tracking
    error_count: int = 0
    timeout_count: int = 0
    success_rate: float = 1.0
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'benchmark_type': self.benchmark_type.value,
            'duration': self.duration,
            'messages_processed': self.messages_processed,
            'messages_per_second': self.messages_per_second,
            'avg_latency_ms': self.avg_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'avg_cpu_percent': self.avg_cpu_percent,
            'max_cpu_percent': self.max_cpu_percent,
            'avg_memory_mb': self.avg_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'success_rate': self.success_rate,
            'metadata': self.metadata
        }


class MessageBusBenchmark:
    """
    Comprehensive benchmarking suite for Enhanced MessageBus.
    
    Provides detailed performance analysis, load testing, and regression detection
    for MessageBus implementations with production-realistic scenarios.
    """
    
    def __init__(self, 
                 messagebus_factory: Callable[[], Any],
                 comparison_messagebus_factory: Optional[Callable[[], Any]] = None):
        """
        Initialize benchmark suite.
        
        Args:
            messagebus_factory: Factory function for creating MessageBus instances
            comparison_messagebus_factory: Factory for comparison/baseline MessageBus
        """
        self.messagebus_factory = messagebus_factory
        self.comparison_factory = comparison_messagebus_factory
        
        # Benchmark state
        self._results_history: Dict[BenchmarkType, List[BenchmarkResult]] = defaultdict(list)
        self._baseline_results: Dict[BenchmarkType, BenchmarkResult] = {}
        
        # Performance monitoring
        self._resource_monitor: Optional[asyncio.Task] = None
        self._resource_data: deque = deque(maxlen=1000)
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run single benchmark with given configuration."""
        self._logger.info(f"Starting {config.benchmark_type.value} benchmark")
        
        # Create MessageBus instance
        messagebus = self.messagebus_factory()
        
        # Start resource monitoring if enabled
        if config.enable_resource_monitoring:
            self._resource_data.clear()
            self._resource_monitor = asyncio.create_task(self._monitor_resources())
        
        # Run benchmark based on type
        benchmark_funcs = {
            BenchmarkType.THROUGHPUT: self._benchmark_throughput,
            BenchmarkType.LATENCY: self._benchmark_latency,
            BenchmarkType.SCALABILITY: self._benchmark_scalability,
            BenchmarkType.RESOURCE_USAGE: self._benchmark_resource_usage,
            BenchmarkType.PRIORITY_FAIRNESS: self._benchmark_priority_fairness,
            BenchmarkType.PATTERN_MATCHING: self._benchmark_pattern_matching,
            BenchmarkType.BATCH_PROCESSING: self._benchmark_batch_processing,
            BenchmarkType.REAL_WORLD_SIMULATION: self._benchmark_real_world
        }
        
        benchmark_func = benchmark_funcs.get(config.benchmark_type)
        if not benchmark_func:
            raise ValueError(f"Unknown benchmark type: {config.benchmark_type}")
        
        try:
            result = await benchmark_func(messagebus, config)
            
            # Store result
            self._results_history[config.benchmark_type].append(result)
            
            self._logger.info(f"Completed {config.benchmark_type.value} benchmark: "
                             f"{result.messages_per_second:.0f} msg/sec, "
                             f"{result.avg_latency_ms:.2f}ms avg latency")
            
            return result
            
        finally:
            # Stop resource monitoring
            if self._resource_monitor:
                self._resource_monitor.cancel()
                try:
                    await self._resource_monitor
                except asyncio.CancelledError:
                    pass
            
            # Cleanup
            if hasattr(messagebus, 'stop'):
                await messagebus.stop()
            elif hasattr(messagebus, 'close'):
                await messagebus.close()
    
    async def _monitor_resources(self) -> None:
        """Monitor system resource usage during benchmarks."""
        process = psutil.Process()
        
        while True:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                self._resource_data.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb
                })
                
                await asyncio.sleep(0.1)  # 100ms sampling
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error monitoring resources: {e}")
                break
    
    async def _benchmark_throughput(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark maximum throughput."""
        start_time = time.time()
        
        # Track messages and latencies
        messages_sent = 0
        messages_received = 0
        latencies = []
        errors = 0
        
        # Message handler
        async def message_handler(topic: str, data: Any) -> None:
            nonlocal messages_received
            messages_received += 1
            
            # Calculate latency if timestamp in message
            if isinstance(data, dict) and 'timestamp' in data:
                latency_ms = (time.time() - data['timestamp']) * 1000
                latencies.append(latency_ms)
        
        # Subscribe to test topics
        if hasattr(messagebus, 'subscribe'):
            await messagebus.subscribe("benchmark.*", message_handler)
        
        # Warmup period
        if config.warmup_seconds > 0:
            await self._run_message_load(messagebus, config, config.warmup_seconds)
            await asyncio.sleep(1)  # Brief pause
            messages_sent = 0
            messages_received = 0
            latencies.clear()
        
        # Main benchmark
        test_start = time.time()
        
        # Create producer tasks
        producer_tasks = []
        for i in range(config.concurrent_producers):
            task = asyncio.create_task(
                self._message_producer(messagebus, config, f"producer_{i}")
            )
            producer_tasks.append(task)
        
        # Run for specified duration
        await asyncio.sleep(config.duration_seconds)
        
        # Stop producers
        for task in producer_tasks:
            task.cancel()
        
        # Wait for remaining messages
        await asyncio.sleep(2)
        
        test_end = time.time()
        actual_duration = test_end - test_start
        
        # Calculate results
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=start_time,
            end_time=test_end,
            duration=actual_duration,
            messages_processed=messages_received,
            messages_per_second=messages_received / actual_duration if actual_duration > 0 else 0,
            error_count=errors
        )
        
        # Calculate latency statistics
        if latencies:
            result.avg_latency_ms = mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            result.p50_latency_ms = np.percentile(latencies, 50)
            result.p95_latency_ms = np.percentile(latencies, 95)
            result.p99_latency_ms = np.percentile(latencies, 99)
        
        # Add resource usage
        if self._resource_data:
            cpu_values = [r['cpu_percent'] for r in self._resource_data]
            memory_values = [r['memory_mb'] for r in self._resource_data]
            
            result.avg_cpu_percent = mean(cpu_values)
            result.max_cpu_percent = max(cpu_values)
            result.avg_memory_mb = mean(memory_values)
            result.max_memory_mb = max(memory_values)
        
        return result
    
    async def _benchmark_latency(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark end-to-end latency."""
        latencies = []
        messages_processed = 0
        
        # Message handler that measures latency
        async def latency_handler(topic: str, data: Any) -> None:
            nonlocal messages_processed
            if isinstance(data, dict) and 'send_time' in data:
                latency_ms = (time.time() - data['send_time']) * 1000
                latencies.append(latency_ms)
                messages_processed += 1
        
        # Subscribe
        if hasattr(messagebus, 'subscribe'):
            await messagebus.subscribe("latency_test", latency_handler)
        
        start_time = time.time()
        
        # Send messages with timestamps
        for i in range(config.message_count):
            message = {
                'id': i,
                'send_time': time.time(),
                'data': 'x' * config.message_size_bytes
            }
            
            if hasattr(messagebus, 'publish'):
                await messagebus.publish("latency_test", message)
            
            # Small delay to prevent overwhelming
            if i % 100 == 0:
                await asyncio.sleep(0.001)
        
        # Wait for all messages to be processed
        timeout = 30  # 30 second timeout
        while messages_processed < config.message_count and timeout > 0:
            await asyncio.sleep(0.1)
            timeout -= 0.1
        
        end_time = time.time()
        
        # Calculate results
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            messages_processed=messages_processed
        )
        
        if latencies:
            result.avg_latency_ms = mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            result.p50_latency_ms = np.percentile(latencies, 50)
            result.p95_latency_ms = np.percentile(latencies, 95)
            result.p99_latency_ms = np.percentile(latencies, 99)
        
        result.success_rate = messages_processed / config.message_count
        
        return result
    
    async def _benchmark_scalability(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark scalability under increasing load."""
        results = []
        
        # Test with increasing number of concurrent producers/consumers
        for concurrency in [1, 2, 4, 8, 16]:
            scale_config = BenchmarkConfig(
                benchmark_type=BenchmarkType.THROUGHPUT,
                duration_seconds=30,  # Shorter duration for scalability test
                concurrent_producers=concurrency,
                concurrent_consumers=concurrency,
                message_count=config.message_count,
                message_size_bytes=config.message_size_bytes
            )
            
            # Run throughput benchmark with current concurrency
            scale_result = await self._benchmark_throughput(messagebus, scale_config)
            results.append({
                'concurrency': concurrency,
                'throughput': scale_result.messages_per_second,
                'latency': scale_result.avg_latency_ms,
                'cpu': scale_result.avg_cpu_percent,
                'memory': scale_result.avg_memory_mb
            })
        
        # Create summary result
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=time.time(),
            end_time=time.time(),
            duration=0,  # Calculated from individual tests
            metadata={'scalability_results': results}
        )
        
        # Calculate aggregate metrics
        if results:
            result.messages_per_second = max(r['throughput'] for r in results)
            result.avg_latency_ms = mean(r['latency'] for r in results)
            result.avg_cpu_percent = mean(r['cpu'] for r in results)
            result.avg_memory_mb = mean(r['memory'] for r in results)
        
        return result
    
    async def _benchmark_resource_usage(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark resource utilization."""
        # Run sustained load and monitor resources
        return await self._benchmark_throughput(messagebus, config)
    
    async def _benchmark_priority_fairness(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark priority queue fairness."""
        priority_counts = defaultdict(int)
        priority_latencies = defaultdict(list)
        
        # Message handler that tracks priority processing
        async def priority_handler(topic: str, data: Any) -> None:
            if isinstance(data, dict) and 'priority' in data and 'send_time' in data:
                priority = MessagePriority(data['priority'])
                latency_ms = (time.time() - data['send_time']) * 1000
                
                priority_counts[priority] += 1
                priority_latencies[priority].append(latency_ms)
        
        # Subscribe
        if hasattr(messagebus, 'subscribe'):
            await messagebus.subscribe("priority_test", priority_handler)
        
        start_time = time.time()
        
        # Send messages with different priorities
        for i in range(config.message_count):
            # Select priority based on distribution
            rand_val = np.random.random()
            cumulative = 0
            priority = MessagePriority.NORMAL
            
            for p, prob in config.priority_distribution.items():
                cumulative += prob
                if rand_val <= cumulative:
                    priority = p
                    break
            
            message = {
                'id': i,
                'priority': priority.value,
                'send_time': time.time(),
                'data': 'x' * config.message_size_bytes
            }
            
            if hasattr(messagebus, 'publish'):
                await messagebus.publish("priority_test", message, priority=priority)
        
        # Wait for processing
        await asyncio.sleep(5)
        end_time = time.time()
        
        # Analyze fairness
        fairness_metrics = {}
        for priority, latencies in priority_latencies.items():
            if latencies:
                fairness_metrics[priority.value] = {
                    'count': len(latencies),
                    'avg_latency_ms': mean(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95)
                }
        
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            messages_processed=sum(priority_counts.values()),
            metadata={'priority_fairness': fairness_metrics}
        )
        
        return result
    
    async def _benchmark_pattern_matching(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark pattern matching performance."""
        # Test various pattern complexities
        patterns = [
            "simple.topic",
            "*.wildcard",
            "multi.*.pattern.*",
            "**",
            "complex.pattern.with.many.levels.*.and.wildcards.**"
        ]
        
        pattern_results = {}
        
        for pattern in patterns:
            matches = 0
            
            # Handler for this pattern
            async def pattern_handler(topic: str, data: Any) -> None:
                nonlocal matches
                matches += 1
            
            # Subscribe to pattern
            if hasattr(messagebus, 'subscribe'):
                await messagebus.subscribe(pattern, pattern_handler)
            
            # Send test messages
            start = time.time()
            for i in range(1000):  # Smaller count for pattern tests
                topic = f"test.topic.{i % 10}.data.{i % 5}"
                if hasattr(messagebus, 'publish'):
                    await messagebus.publish(topic, {'id': i})
            
            await asyncio.sleep(1)  # Wait for processing
            end = time.time()
            
            pattern_results[pattern] = {
                'matches': matches,
                'duration': end - start,
                'matches_per_second': matches / (end - start) if end - start > 0 else 0
            }
        
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=time.time(),
            end_time=time.time(),
            duration=0,
            metadata={'pattern_results': pattern_results}
        )
        
        return result
    
    async def _benchmark_batch_processing(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark batch processing efficiency."""
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100, 500, 1000]
        batch_results = {}
        
        for batch_size in batch_sizes:
            messages_processed = 0
            start_time = time.time()
            
            # Handler
            async def batch_handler(topic: str, data: Any) -> None:
                nonlocal messages_processed
                if isinstance(data, list):
                    messages_processed += len(data)
                else:
                    messages_processed += 1
            
            # Subscribe
            if hasattr(messagebus, 'subscribe'):
                await messagebus.subscribe(f"batch_{batch_size}", batch_handler)
            
            # Send batches
            total_messages = 10000
            num_batches = total_messages // batch_size
            
            for batch_id in range(num_batches):
                batch = [{'id': i, 'data': 'x' * 100} for i in range(batch_size)]
                
                if hasattr(messagebus, 'publish_batch'):
                    await messagebus.publish_batch(f"batch_{batch_size}", batch)
                else:
                    # Simulate batch by sending individual messages quickly
                    for msg in batch:
                        if hasattr(messagebus, 'publish'):
                            await messagebus.publish(f"batch_{batch_size}", msg)
            
            # Wait for processing
            await asyncio.sleep(2)
            end_time = time.time()
            
            duration = end_time - start_time
            batch_results[batch_size] = {
                'messages_processed': messages_processed,
                'duration': duration,
                'messages_per_second': messages_processed / duration if duration > 0 else 0
            }
        
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=time.time(),
            end_time=time.time(),
            duration=0,
            metadata={'batch_results': batch_results}
        )
        
        return result
    
    async def _benchmark_real_world(self, messagebus: Any, config: BenchmarkConfig) -> BenchmarkResult:
        """Benchmark real-world trading scenario."""
        # Simulate realistic trading message patterns
        scenarios = {
            'market_data': {'rate': 1000, 'priority': MessagePriority.HIGH},
            'order_updates': {'rate': 100, 'priority': MessagePriority.CRITICAL},
            'portfolio_updates': {'rate': 50, 'priority': MessagePriority.NORMAL},
            'risk_alerts': {'rate': 10, 'priority': MessagePriority.CRITICAL},
            'analytics': {'rate': 25, 'priority': MessagePriority.LOW}
        }
        
        total_messages = 0
        scenario_results = {}
        
        # Handlers for each scenario
        for scenario_name, scenario_config in scenarios.items():
            messages_received = 0
            latencies = []
            
            async def scenario_handler(topic: str, data: Any, scenario=scenario_name) -> None:
                nonlocal messages_received
                messages_received += 1
                if isinstance(data, dict) and 'timestamp' in data:
                    latency_ms = (time.time() - data['timestamp']) * 1000
                    latencies.append(latency_ms)
            
            if hasattr(messagebus, 'subscribe'):
                await messagebus.subscribe(f"trading.{scenario_name}.*", scenario_handler)
        
        start_time = time.time()
        
        # Run realistic load for duration
        producer_tasks = []
        for scenario_name, scenario_config in scenarios.items():
            task = asyncio.create_task(
                self._realistic_producer(
                    messagebus, scenario_name, scenario_config, config.duration_seconds
                )
            )
            producer_tasks.append(task)
        
        # Wait for completion
        await asyncio.gather(*producer_tasks)
        await asyncio.sleep(2)  # Wait for processing
        
        end_time = time.time()
        
        result = BenchmarkResult(
            benchmark_type=config.benchmark_type,
            config=config,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            metadata={'scenario_results': scenario_results}
        )
        
        return result
    
    async def _message_producer(self, messagebus: Any, config: BenchmarkConfig, producer_id: str) -> None:
        """Generic message producer for benchmarks."""
        message_id = 0
        
        while True:
            try:
                # Create message with timestamp
                message = {
                    'producer_id': producer_id,
                    'message_id': message_id,
                    'timestamp': time.time(),
                    'data': 'x' * config.message_size_bytes
                }
                
                # Select priority
                priority = self._select_priority(config.priority_distribution)
                
                # Publish message
                if hasattr(messagebus, 'publish'):
                    await messagebus.publish(f"benchmark.{producer_id}", message, priority=priority)
                
                message_id += 1
                
                # Apply load profile delay
                delay = self._calculate_producer_delay(config)
                if delay > 0:
                    await asyncio.sleep(delay)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in producer {producer_id}: {e}")
                break
    
    async def _realistic_producer(self, messagebus: Any, scenario: str, 
                                scenario_config: Dict[str, Any], duration: int) -> None:
        """Produce realistic trading messages."""
        start_time = time.time()
        message_id = 0
        
        while time.time() - start_time < duration:
            try:
                # Create scenario-specific message
                if scenario == 'market_data':
                    message = {
                        'type': 'tick',
                        'symbol': f"SYMBOL_{message_id % 10}",
                        'price': 100.0 + (message_id % 100) / 100.0,
                        'volume': 1000 + (message_id % 500),
                        'timestamp': time.time()
                    }
                elif scenario == 'order_updates':
                    message = {
                        'type': 'order_update',
                        'order_id': f"ORDER_{message_id}",
                        'status': 'FILLED' if message_id % 3 == 0 else 'PARTIAL',
                        'quantity': 100 + (message_id % 50),
                        'timestamp': time.time()
                    }
                else:
                    message = {
                        'type': scenario,
                        'id': message_id,
                        'timestamp': time.time(),
                        'data': f"Data for {scenario}"
                    }
                
                # Publish with appropriate priority
                priority = scenario_config['priority']
                if hasattr(messagebus, 'publish'):
                    await messagebus.publish(f"trading.{scenario}.update", message, priority=priority)
                
                message_id += 1
                
                # Rate limiting based on scenario
                rate = scenario_config['rate']
                await asyncio.sleep(1.0 / rate)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in realistic producer {scenario}: {e}")
                break
    
    def _select_priority(self, distribution: Dict[MessagePriority, float]) -> MessagePriority:
        """Select message priority based on distribution."""
        rand_val = np.random.random()
        cumulative = 0
        
        for priority, prob in distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                return priority
        
        return MessagePriority.NORMAL
    
    def _calculate_producer_delay(self, config: BenchmarkConfig) -> float:
        """Calculate delay between messages based on load profile."""
        if config.load_profile == LoadProfile.CONSTANT:
            return 0.001  # 1ms constant delay
        elif config.load_profile == LoadProfile.BURSTY:
            # Burst every 100 messages
            return 0.001 if np.random.random() > 0.1 else 0.01
        elif config.load_profile == LoadProfile.SPIKE:
            # Occasional spikes
            return 0.001 if np.random.random() > 0.05 else 0.1
        else:
            return 0.001
    
    async def run_comparison_benchmark(self, config: BenchmarkConfig) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Run comparison benchmark between enhanced and standard MessageBus."""
        if not self.comparison_factory:
            raise ValueError("Comparison factory not provided")
        
        # Run enhanced MessageBus benchmark
        enhanced_result = await self.run_benchmark(config)
        
        # Run comparison MessageBus benchmark
        comparison_messagebus = self.comparison_factory()
        
        try:
            # Use same config for fair comparison
            comparison_config = BenchmarkConfig(
                benchmark_type=config.benchmark_type,
                duration_seconds=config.duration_seconds,
                message_count=config.message_count,
                concurrent_producers=config.concurrent_producers,
                concurrent_consumers=config.concurrent_consumers,
                message_size_bytes=config.message_size_bytes
            )
            
            comparison_result = await self._benchmark_throughput(comparison_messagebus, comparison_config)
            comparison_result.benchmark_type = config.benchmark_type
            
        finally:
            if hasattr(comparison_messagebus, 'stop'):
                await comparison_messagebus.stop()
            elif hasattr(comparison_messagebus, 'close'):
                await comparison_messagebus.close()
        
        return enhanced_result, comparison_result
    
    def analyze_performance_regression(self, current_result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze performance regression against historical baselines."""
        benchmark_type = current_result.benchmark_type
        
        if benchmark_type not in self._baseline_results:
            # No baseline, set current as baseline
            self._baseline_results[benchmark_type] = current_result
            return {'status': 'baseline_set', 'regression_detected': False}
        
        baseline = self._baseline_results[benchmark_type]
        
        # Calculate performance changes
        throughput_change = (
            (current_result.messages_per_second - baseline.messages_per_second) /
            baseline.messages_per_second * 100 if baseline.messages_per_second > 0 else 0
        )
        
        latency_change = (
            (current_result.avg_latency_ms - baseline.avg_latency_ms) /
            baseline.avg_latency_ms * 100 if baseline.avg_latency_ms > 0 else 0
        )
        
        cpu_change = (
            (current_result.avg_cpu_percent - baseline.avg_cpu_percent) /
            baseline.avg_cpu_percent * 100 if baseline.avg_cpu_percent > 0 else 0
        )
        
        memory_change = (
            (current_result.avg_memory_mb - baseline.avg_memory_mb) /
            baseline.avg_memory_mb * 100 if baseline.avg_memory_mb > 0 else 0
        )
        
        # Detect regressions (thresholds)
        regression_detected = (
            throughput_change < -10.0 or  # 10% throughput drop
            latency_change > 20.0 or      # 20% latency increase
            cpu_change > 25.0 or          # 25% CPU increase
            memory_change > 30.0          # 30% memory increase
        )
        
        return {
            'regression_detected': regression_detected,
            'throughput_change_percent': throughput_change,
            'latency_change_percent': latency_change,
            'cpu_change_percent': cpu_change,
            'memory_change_percent': memory_change,
            'baseline_date': baseline.start_time,
            'current_date': current_result.start_time
        }
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        summary = {}
        
        for benchmark_type, results in self._results_history.items():
            if results:
                latest = results[-1]
                summary[benchmark_type.value] = {
                    'total_runs': len(results),
                    'latest_result': latest.to_dict(),
                    'avg_throughput': mean(r.messages_per_second for r in results),
                    'avg_latency': mean(r.avg_latency_ms for r in results)
                }
        
        return summary
    
    def export_results(self, filepath: str) -> None:
        """Export benchmark results to JSON file."""
        import json
        
        data = {
            'results_history': {},
            'baselines': {},
            'export_timestamp': time.time()
        }
        
        # Convert results to serializable format
        for benchmark_type, results in self._results_history.items():
            data['results_history'][benchmark_type.value] = [
                r.to_dict() for r in results
            ]
        
        for benchmark_type, baseline in self._baseline_results.items():
            data['baselines'][benchmark_type.value] = baseline.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._logger.info(f"Exported benchmark results to {filepath}")

# Utility functions for common benchmark scenarios

async def quick_throughput_test(messagebus_factory: Callable[[], Any]) -> BenchmarkResult:
    """Run quick throughput test."""
    benchmark = MessageBusBenchmark(messagebus_factory)
    config = BenchmarkConfig(
        benchmark_type=BenchmarkType.THROUGHPUT,
        duration_seconds=30,
        message_count=10000
    )
    return await benchmark.run_benchmark(config)

async def comprehensive_benchmark_suite(messagebus_factory: Callable[[], Any]) -> Dict[str, BenchmarkResult]:
    """Run comprehensive benchmark suite."""
    benchmark = MessageBusBenchmark(messagebus_factory)
    results = {}
    
    # Define test configurations
    test_configs = [
        BenchmarkConfig(BenchmarkType.THROUGHPUT, duration_seconds=60),
        BenchmarkConfig(BenchmarkType.LATENCY, message_count=10000),
        BenchmarkConfig(BenchmarkType.SCALABILITY, duration_seconds=30),
        BenchmarkConfig(BenchmarkType.PRIORITY_FAIRNESS, message_count=5000),
        BenchmarkConfig(BenchmarkType.REAL_WORLD_SIMULATION, duration_seconds=60)
    ]
    
    for config in test_configs:
        result = await benchmark.run_benchmark(config)
        results[config.benchmark_type.value] = result
    
    return results
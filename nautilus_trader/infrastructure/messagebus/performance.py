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

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from statistics import mean, median, stdev
import psutil
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .messagebus_config_enhanced import MessagePriority, EnhancedMessageBusConfig
from .enhanced_messagebus_client import BufferedMessageBusClient
from .enhanced_redis_streams import RedisStreamManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    total_messages: int = 0
    messages_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    buffer_utilization: float = 0.0
    throughput_mbps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    duration_seconds: int = 60
    message_rate: int = 1000  # messages per second
    message_size: int = 1024  # bytes
    pattern_complexity: int = 5  # number of different topic patterns
    concurrent_producers: int = 5
    concurrent_consumers: int = 5
    warmup_duration: int = 10  # seconds
    collect_individual_latencies: bool = True
    memory_profiling: bool = True

class TopicGenerator:
    """Generate topics similar to NautilusTrader patterns"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.categories = ["data", "events", "orders", "trades", "market"]
        self.models = ["quotes", "trades", "bars", "orderbooks", "depths", "tickers"]
        self.venues = ["BINANCE", "BYBIT", "OKX", "KRAKEN", "COINBASE", "FTX"]
        self.instruments = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
        self.priorities = ["normal", "high", "critical", "low"]
    
    def generate_topics(self, n: int) -> List[str]:
        """Generate n random topics"""
        topics = []
        for _ in range(n):
            cat = random.choice(self.categories)
            model = random.choice(self.models)
            venue = random.choice(self.venues)
            instrument = random.choice(self.instruments)
            priority = random.choice(self.priorities)
            topics.append(f"{cat}.{model}.{venue}.{instrument}.{priority}")
        return topics
    
    def generate_pattern(self) -> str:
        """Generate a topic pattern for matching"""
        patterns = [
            "data.*.BINANCE.*",
            "events.trades.*.*",
            "orders.*.*.BTCUSDT.*",
            "*.quotes.*.*",
            "market.*.*.*critical",
            "data.bars.*.*",
        ]
        return random.choice(patterns)

class LatencyTracker:
    """Track message latencies with high precision"""
    
    def __init__(self, max_samples: int = 100000):
        self.latencies: deque = deque(maxlen=max_samples)
        self.sent_times: Dict[str, float] = {}
        self.error_count = 0
        self.total_messages = 0
        
    def mark_sent(self, message_id: str):
        """Mark message as sent"""
        self.sent_times[message_id] = time.perf_counter()
        self.total_messages += 1
    
    def mark_received(self, message_id: str):
        """Mark message as received and calculate latency"""
        if message_id in self.sent_times:
            latency = (time.perf_counter() - self.sent_times[message_id]) * 1000  # ms
            self.latencies.append(latency)
            del self.sent_times[message_id]
        else:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {
                "avg": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0,
                "min": 0.0, "stdev": 0.0, "count": 0
            }
        
        sorted_latencies = sorted(self.latencies)
        count = len(sorted_latencies)
        
        return {
            "avg": mean(sorted_latencies),
            "median": median(sorted_latencies),
            "p95": sorted_latencies[int(count * 0.95)] if count > 20 else sorted_latencies[-1],
            "p99": sorted_latencies[int(count * 0.99)] if count > 100 else sorted_latencies[-1],
            "max": max(sorted_latencies),
            "min": min(sorted_latencies),
            "stdev": stdev(sorted_latencies) if count > 1 else 0.0,
            "count": count
        }

class MessageBusBenchmark:
    """Comprehensive MessageBus benchmark suite"""
    
    def __init__(self, config: BenchmarkConfig, messagebus_config: EnhancedMessageBusConfig):
        self.config = config
        self.messagebus_config = messagebus_config
        self.topic_generator = TopicGenerator()
        self.latency_tracker = LatencyTracker()
        self.metrics_history: List[PerformanceMetrics] = []
        self.running = False
        
        # Initialize components
        self.messagebus_client = BufferedMessageBusClient(messagebus_config)
        self.stream_manager = RedisStreamManager(messagebus_config)
    
    async def generate_test_message(self, topic: str, message_id: str) -> bytes:
        """Generate a test message with specified size"""
        base_message = {
            "id": message_id,
            "topic": topic,
            "timestamp": time.time(),
            "priority": random.choice(["normal", "high", "critical"]),
            "source": "benchmark",
        }
        
        # Pad message to reach target size
        padding_size = max(0, self.config.message_size - len(str(base_message)))
        base_message["padding"] = "x" * padding_size
        
        return str(base_message).encode('utf-8')
    
    async def producer_worker(self, worker_id: int, topics: List[str], duration: float):
        """Producer worker that sends messages"""
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < duration and self.running:
            topic = random.choice(topics)
            message_id = f"producer_{worker_id}_{message_count}_{time.time()}"
            message = await self.generate_test_message(topic, message_id)
            
            try:
                # Mark as sent for latency tracking
                self.latency_tracker.mark_sent(message_id)
                
                # Send via messagebus
                await self.messagebus_client.publish(topic, message)
                message_count += 1
                
                # Rate limiting
                if self.config.message_rate > 0:
                    await asyncio.sleep(1.0 / self.config.message_rate)
                    
            except Exception as e:
                logger.error(f"Producer {worker_id} error: {e}")
                self.latency_tracker.error_count += 1
        
        logger.info(f"Producer {worker_id} sent {message_count} messages")
    
    async def consumer_worker(self, worker_id: int, patterns: List[str], duration: float):
        """Consumer worker that receives messages"""
        start_time = time.time()
        message_count = 0
        
        # Subscribe to patterns
        for pattern in patterns:
            await self.messagebus_client.subscribe(pattern)
        
        while time.time() - start_time < duration and self.running:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(
                    self.messagebus_client.receive(),
                    timeout=1.0
                )
                
                if message:
                    # Extract message ID for latency tracking
                    try:
                        import json
                        msg_data = json.loads(message.decode('utf-8'))
                        if 'id' in msg_data:
                            self.latency_tracker.mark_received(msg_data['id'])
                    except:
                        pass  # Skip if message format is unexpected
                    
                    message_count += 1
                    
            except asyncio.TimeoutError:
                continue  # No message received, continue
            except Exception as e:
                logger.error(f"Consumer {worker_id} error: {e}")
        
        logger.info(f"Consumer {worker_id} received {message_count} messages")
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics"""
        process = psutil.Process()
        
        try:
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            # Get Redis memory usage if possible
            redis_memory = 0.0
            try:
                redis_info = await self.stream_manager.redis_client.info('memory')
                redis_memory = redis_info.get('used_memory', 0) / (1024 * 1024)  # MB
            except:
                pass
            
            return {
                "memory_mb": memory_info.rss / (1024 * 1024),
                "redis_memory_mb": redis_memory,
                "cpu_percent": cpu_percent,
                "buffer_size": len(self.messagebus_client._message_buffer) if hasattr(self.messagebus_client, '_message_buffer') else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"memory_mb": 0, "redis_memory_mb": 0, "cpu_percent": 0, "buffer_size": 0}
    
    async def run_pattern_matching_benchmark(self) -> Dict[str, Any]:
        """Benchmark topic pattern matching performance"""
        topics = self.topic_generator.generate_topics(10000)
        patterns = [self.topic_generator.generate_pattern() for _ in range(100)]
        
        start_time = time.perf_counter()
        total_matches = 0
        
        for topic in topics:
            for pattern in patterns:
                # Simulate pattern matching (would use actual implementation)
                if self._simple_pattern_match(pattern, topic):
                    total_matches += 1
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        return {
            "total_operations": len(topics) * len(patterns),
            "total_matches": total_matches,
            "duration_ms": duration * 1000,
            "operations_per_second": (len(topics) * len(patterns)) / duration,
            "match_rate": total_matches / (len(topics) * len(patterns))
        }
    
    def _simple_pattern_match(self, pattern: str, topic: str) -> bool:
        """Simple pattern matching for benchmarking"""
        import re
        regex_pattern = pattern.replace('.', r'\.')
        regex_pattern = regex_pattern.replace('*', '.*')
        return bool(re.match(f"^{regex_pattern}$", topic))
    
    async def run_throughput_benchmark(self) -> PerformanceMetrics:
        """Run comprehensive throughput benchmark"""
        logger.info("Starting throughput benchmark...")
        
        # Generate test data
        topics = self.topic_generator.generate_topics(self.config.pattern_complexity * 10)
        patterns = [self.topic_generator.generate_pattern() for _ in range(self.config.pattern_complexity)]
        
        # Warmup phase
        logger.info(f"Warmup phase: {self.config.warmup_duration} seconds")
        self.running = True
        warmup_tasks = []
        
        # Start fewer workers for warmup
        for i in range(2):
            warmup_tasks.append(
                asyncio.create_task(self.producer_worker(i, topics, self.config.warmup_duration))
            )
            warmup_tasks.append(
                asyncio.create_task(self.consumer_worker(i, patterns, self.config.warmup_duration))
            )
        
        await asyncio.gather(*warmup_tasks)
        
        # Reset tracking after warmup
        self.latency_tracker = LatencyTracker()
        
        # Main benchmark phase
        logger.info(f"Main benchmark phase: {self.config.duration_seconds} seconds")
        start_time = time.time()
        
        # Start all workers
        tasks = []
        for i in range(self.config.concurrent_producers):
            tasks.append(
                asyncio.create_task(self.producer_worker(i, topics, self.config.duration_seconds))
            )
        
        for i in range(self.config.concurrent_consumers):
            tasks.append(
                asyncio.create_task(self.consumer_worker(i, patterns, self.config.duration_seconds))
            )
        
        # Run benchmark
        await asyncio.gather(*tasks)
        
        # Collect final metrics
        end_time = time.time()
        duration = end_time - start_time
        
        latency_stats = self.latency_tracker.get_stats()
        system_metrics = await self.collect_system_metrics()
        
        # Calculate throughput
        total_bytes = self.latency_tracker.total_messages * self.config.message_size
        throughput_mbps = (total_bytes / (1024 * 1024)) / duration
        
        metrics = PerformanceMetrics(
            total_messages=self.latency_tracker.total_messages,
            messages_per_second=self.latency_tracker.total_messages / duration,
            avg_latency_ms=latency_stats["avg"],
            median_latency_ms=latency_stats["median"],
            p95_latency_ms=latency_stats["p95"],
            p99_latency_ms=latency_stats["p99"],
            max_latency_ms=latency_stats["max"],
            error_rate=self.latency_tracker.error_count / max(1, self.latency_tracker.total_messages),
            memory_usage_mb=system_metrics["memory_mb"],
            cpu_usage_percent=system_metrics["cpu_percent"],
            buffer_utilization=system_metrics["buffer_size"] / max(1, self.messagebus_config.default_buffer_config.max_size),
            throughput_mbps=throughput_mbps
        )
        
        self.metrics_history.append(metrics)
        self.running = False
        
        logger.info("Throughput benchmark completed")
        return metrics
    
    async def run_stress_test(self, max_rate: int = 100000) -> Dict[str, PerformanceMetrics]:
        """Run stress test with increasing load"""
        logger.info("Starting stress test...")
        
        results = {}
        rates = [1000, 5000, 10000, 25000, 50000, max_rate]
        
        for rate in rates:
            if not self.running:
                break
            
            logger.info(f"Testing rate: {rate} msg/sec")
            
            # Update config for this rate
            old_rate = self.config.message_rate
            old_duration = self.config.duration_seconds
            
            self.config.message_rate = rate
            self.config.duration_seconds = 30  # Shorter duration for stress test
            
            try:
                metrics = await self.run_throughput_benchmark()
                results[f"{rate}_msg_sec"] = metrics
                
                # Check if we're hitting limits
                if metrics.error_rate > 0.05:  # 5% error rate
                    logger.warning(f"High error rate at {rate} msg/sec: {metrics.error_rate:.2%}")
                    break
                
            except Exception as e:
                logger.error(f"Stress test failed at {rate} msg/sec: {e}")
                break
            finally:
                # Restore config
                self.config.message_rate = old_rate
                self.config.duration_seconds = old_duration
                
                # Cool down between tests
                await asyncio.sleep(5)
        
        return results
    
    def generate_report(self, metrics: PerformanceMetrics, pattern_results: Dict = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            "summary": {
                "total_messages": metrics.total_messages,
                "duration_seconds": self.config.duration_seconds,
                "messages_per_second": round(metrics.messages_per_second, 2),
                "throughput_mbps": round(metrics.throughput_mbps, 2),
                "error_rate_percent": round(metrics.error_rate * 100, 2),
            },
            "latency": {
                "average_ms": round(metrics.avg_latency_ms, 2),
                "median_ms": round(metrics.median_latency_ms, 2),
                "p95_ms": round(metrics.p95_latency_ms, 2),
                "p99_ms": round(metrics.p99_latency_ms, 2),
                "max_ms": round(metrics.max_latency_ms, 2),
            },
            "resources": {
                "memory_usage_mb": round(metrics.memory_usage_mb, 2),
                "cpu_usage_percent": round(metrics.cpu_usage_percent, 2),
                "buffer_utilization_percent": round(metrics.buffer_utilization * 100, 2),
            },
            "configuration": {
                "concurrent_producers": self.config.concurrent_producers,
                "concurrent_consumers": self.config.concurrent_consumers,
                "message_size_bytes": self.config.message_size,
                "target_rate_msg_sec": self.config.message_rate,
                "pattern_complexity": self.config.pattern_complexity,
            }
        }
        
        if pattern_results:
            report["pattern_matching"] = pattern_results
        
        return report

# Convenience functions for running benchmarks
async def run_quick_benchmark() -> Dict[str, Any]:
    """Run a quick benchmark with default settings"""
    config = BenchmarkConfig(duration_seconds=30, message_rate=1000)
    messagebus_config = EnhancedMessageBusConfig()
    
    benchmark = MessageBusBenchmark(config, messagebus_config)
    
    try:
        metrics = await benchmark.run_throughput_benchmark()
        pattern_results = await benchmark.run_pattern_matching_benchmark()
        return benchmark.generate_report(metrics, pattern_results)
    finally:
        await benchmark.messagebus_client.close()
        await benchmark.stream_manager.close()

async def run_production_benchmark() -> Dict[str, Any]:
    """Run production-grade benchmark"""
    config = BenchmarkConfig(
        duration_seconds=300,  # 5 minutes
        message_rate=10000,
        concurrent_producers=10,
        concurrent_consumers=10,
        pattern_complexity=20
    )
    
    from .messagebus_config_enhanced import ConfigPresets
    messagebus_config = ConfigPresets.production()
    
    benchmark = MessageBusBenchmark(config, messagebus_config)
    
    try:
        # Run comprehensive tests
        throughput_metrics = await benchmark.run_throughput_benchmark()
        pattern_results = await benchmark.run_pattern_matching_benchmark()
        stress_results = await benchmark.run_stress_test()
        
        report = benchmark.generate_report(throughput_metrics, pattern_results)
        report["stress_test"] = {
            rate: {
                "messages_per_second": stress_results[rate].messages_per_second,
                "avg_latency_ms": stress_results[rate].avg_latency_ms,
                "error_rate": stress_results[rate].error_rate
            }
            for rate in stress_results
        }
        
        return report
    finally:
        await benchmark.messagebus_client.close()
        await benchmark.stream_manager.close()
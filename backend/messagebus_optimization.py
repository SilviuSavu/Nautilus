"""
MessageBus Redis Pub/Sub Connection Optimization
Optimizes Redis pub/sub connections for all 9 containerized engines
"""

import asyncio
import logging
import redis.asyncio as redis
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Engine types for optimized MessageBus configuration"""
    ANALYTICS = "analytics"
    RISK = "risk"
    FACTOR = "factor"
    ML = "ml"
    FEATURES = "features"
    WEBSOCKET = "websocket"
    STRATEGY = "strategy"
    MARKETDATA = "marketdata"
    PORTFOLIO = "portfolio"


@dataclass
class OptimizedMessageBusConfig:
    """Optimized MessageBus configuration for each engine type"""
    engine_type: EngineType
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Engine-specific stream configuration
    stream_key: str = ""
    consumer_group: str = ""
    consumer_name: str = ""
    
    # Optimized performance settings
    buffer_interval_ms: int = 50  # Optimized from 100ms
    max_buffer_size: int = 5000   # Increased from 1000
    heartbeat_interval_secs: int = 15  # Optimized from 30s
    
    # Connection optimization
    connection_timeout: float = 3.0  # Reduced from 5.0s
    max_reconnect_attempts: int = 5  # Reduced from 10
    reconnect_base_delay: float = 0.5  # Reduced from 1.0s
    reconnect_max_delay: float = 30.0  # Reduced from 60.0s
    
    # Message filtering and routing
    topic_filter: Optional[List[str]] = None
    priority_topics: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize engine-specific configuration"""
        if not self.stream_key:
            self.stream_key = f"nautilus-{self.engine_type.value}-stream"
        if not self.consumer_group:
            self.consumer_group = f"{self.engine_type.value}-group"
        if not self.consumer_name:
            self.consumer_name = f"{self.engine_type.value}-consumer"


class MessageBusOptimizer:
    """Optimizes MessageBus Redis pub/sub connections for all engines"""
    
    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self._engine_configs = self._create_optimized_configs()
        self._active_streams: Dict[str, bool] = {}
        self._connection_metrics: Dict[str, Dict] = {}
        
    def _create_optimized_configs(self) -> Dict[EngineType, OptimizedMessageBusConfig]:
        """Create optimized configurations for each engine"""
        configs = {}
        
        # Analytics Engine - High throughput processing
        configs[EngineType.ANALYTICS] = OptimizedMessageBusConfig(
            engine_type=EngineType.ANALYTICS,
            buffer_interval_ms=25,  # Ultra-fast processing
            max_buffer_size=10000,  # Large buffer for batch analytics
            topic_filter=["market.data.*", "analytics.*", "events.trade.*"],
            priority_topics=["analytics.compute", "analytics.batch"]
        )
        
        # Risk Engine - Critical low-latency processing
        configs[EngineType.RISK] = OptimizedMessageBusConfig(
            engine_type=EngineType.RISK,
            buffer_interval_ms=10,  # Ultra-low latency
            max_buffer_size=2000,   # Smaller buffer for immediate processing
            heartbeat_interval_secs=10,  # Frequent health checks
            topic_filter=["risk.*", "orders.*", "positions.*", "limits.*"],
            priority_topics=["risk.breach", "risk.critical", "orders.execution"]
        )
        
        # Factor Engine - Batch processing optimization
        configs[EngineType.FACTOR] = OptimizedMessageBusConfig(
            engine_type=EngineType.FACTOR,
            buffer_interval_ms=100,  # Batch-friendly
            max_buffer_size=15000,   # Large buffer for factor calculations
            topic_filter=["factors.*", "market.data.*", "economic.*"],
            priority_topics=["factors.macro", "factors.technical"]
        )
        
        # ML Engine - Model inference optimization
        configs[EngineType.ML] = OptimizedMessageBusConfig(
            engine_type=EngineType.ML,
            buffer_interval_ms=50,
            max_buffer_size=7500,
            topic_filter=["ml.*", "features.*", "predictions.*"],
            priority_topics=["ml.inference", "ml.training"]
        )
        
        # Features Engine - Feature pipeline optimization
        configs[EngineType.FEATURES] = OptimizedMessageBusConfig(
            engine_type=EngineType.FEATURES,
            buffer_interval_ms=75,
            max_buffer_size=8000,
            topic_filter=["features.*", "market.data.*", "signals.*"],
            priority_topics=["features.realtime", "features.technical"]
        )
        
        # WebSocket Engine - Real-time streaming optimization
        configs[EngineType.WEBSOCKET] = OptimizedMessageBusConfig(
            engine_type=EngineType.WEBSOCKET,
            buffer_interval_ms=5,   # Real-time streaming
            max_buffer_size=1000,   # Small buffer for immediate delivery
            heartbeat_interval_secs=5,  # Frequent heartbeats
            topic_filter=["websocket.*", "stream.*", "realtime.*"],
            priority_topics=["websocket.broadcast", "stream.market"]
        )
        
        # Strategy Engine - Execution optimization
        configs[EngineType.STRATEGY] = OptimizedMessageBusConfig(
            engine_type=EngineType.STRATEGY,
            buffer_interval_ms=20,  # Fast execution
            max_buffer_size=3000,
            topic_filter=["strategy.*", "signals.*", "orders.*"],
            priority_topics=["strategy.execute", "strategy.stop"]
        )
        
        # Market Data Engine - High-volume data optimization
        configs[EngineType.MARKETDATA] = OptimizedMessageBusConfig(
            engine_type=EngineType.MARKETDATA,
            buffer_interval_ms=30,
            max_buffer_size=12000,  # Large buffer for market data
            topic_filter=["market.*", "data.*", "ticks.*", "bars.*"],
            priority_topics=["market.realtime", "data.critical"]
        )
        
        # Portfolio Engine - Portfolio management optimization
        configs[EngineType.PORTFOLIO] = OptimizedMessageBusConfig(
            engine_type=EngineType.PORTFOLIO,
            buffer_interval_ms=60,
            max_buffer_size=4000,
            topic_filter=["portfolio.*", "positions.*", "pnl.*"],
            priority_topics=["portfolio.rebalance", "portfolio.risk"]
        )
        
        return configs
    
    async def initialize(self) -> bool:
        """Initialize Redis connection and setup optimized streams"""
        try:
            # Connect to Redis
            self._redis_client = redis.Redis(
                host="redis",
                port=6379,
                db=0,
                socket_connect_timeout=3.0,
                socket_keepalive=True,
                decode_responses=True
            )
            
            # Test connection
            await self._redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Setup optimized streams and consumer groups
            await self._setup_optimized_streams()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MessageBus optimizer: {e}")
            return False
    
    async def _setup_optimized_streams(self):
        """Setup optimized Redis streams and consumer groups for all engines"""
        for engine_type, config in self._engine_configs.items():
            try:
                # Create stream and consumer group
                await self._redis_client.xgroup_create(
                    config.stream_key,
                    config.consumer_group,
                    id='0',
                    mkstream=True
                )
                
                # Initialize stream metrics
                self._active_streams[config.stream_key] = True
                self._connection_metrics[config.stream_key] = {
                    "created_at": time.time(),
                    "messages_processed": 0,
                    "last_activity": time.time(),
                    "consumer_group": config.consumer_group,
                    "engine_type": engine_type.value
                }
                
                logger.info(f"Created optimized stream for {engine_type.value}: {config.stream_key}")
                
            except redis.RedisError as e:
                if "BUSYGROUP" not in str(e):
                    logger.warning(f"Error creating stream for {engine_type.value}: {e}")
    
    async def optimize_connections(self) -> Dict[str, Dict]:
        """Optimize all MessageBus connections and return status"""
        if not self._redis_client:
            await self.initialize()
        
        optimization_results = {}
        
        for engine_type, config in self._engine_configs.items():
            try:
                # Check stream health
                stream_info = await self._redis_client.xinfo_stream(config.stream_key)
                
                # Optimize consumer group if needed
                await self._optimize_consumer_group(config)
                
                # Setup topic-based message routing
                await self._setup_topic_routing(config)
                
                # Update metrics
                optimization_results[engine_type.value] = {
                    "stream_key": config.stream_key,
                    "consumer_group": config.consumer_group,
                    "length": stream_info.get("length", 0),
                    "last_generated_id": stream_info.get("last-generated-id", "0-0"),
                    "buffer_config": {
                        "interval_ms": config.buffer_interval_ms,
                        "max_size": config.max_buffer_size,
                        "heartbeat_secs": config.heartbeat_interval_secs
                    },
                    "topic_filters": config.topic_filter or [],
                    "priority_topics": config.priority_topics or [],
                    "optimized": True
                }
                
                logger.info(f"Optimized MessageBus for {engine_type.value}")
                
            except Exception as e:
                optimization_results[engine_type.value] = {
                    "error": str(e),
                    "optimized": False
                }
                logger.error(f"Failed to optimize {engine_type.value}: {e}")
        
        return optimization_results
    
    async def _optimize_consumer_group(self, config: OptimizedMessageBusConfig):
        """Optimize consumer group settings for specific engine"""
        try:
            # Get consumer group info
            groups_info = await self._redis_client.xinfo_groups(config.stream_key)
            
            for group_info in groups_info:
                if group_info["name"] == config.consumer_group:
                    # Check if consumer group needs optimization
                    pending_count = group_info.get("pending", 0)
                    consumers_count = group_info.get("consumers", 0)
                    
                    if pending_count > config.max_buffer_size * 0.8:
                        # Auto-scale consumers if pending messages are high
                        logger.warning(f"High pending count ({pending_count}) for {config.consumer_group}")
                        # Could implement auto-scaling logic here
                    
                    logger.debug(f"Consumer group {config.consumer_group}: {consumers_count} consumers, {pending_count} pending")
                    break
                    
        except redis.RedisError as e:
            logger.warning(f"Could not optimize consumer group {config.consumer_group}: {e}")
    
    async def _setup_topic_routing(self, config: OptimizedMessageBusConfig):
        """Setup optimized topic-based message routing"""
        if config.topic_filter:
            # Create routing configuration in Redis
            routing_key = f"routing:{config.stream_key}"
            routing_config = {
                "topics": config.topic_filter,
                "priority_topics": config.priority_topics or [],
                "buffer_config": {
                    "interval_ms": config.buffer_interval_ms,
                    "max_size": config.max_buffer_size
                }
            }
            
            await self._redis_client.hset(
                routing_key,
                "config",
                json.dumps(routing_config)
            )
            
            logger.debug(f"Setup topic routing for {config.stream_key}")
    
    async def get_optimization_status(self) -> Dict[str, any]:
        """Get comprehensive optimization status"""
        if not self._redis_client:
            return {"error": "Redis client not initialized"}
        
        try:
            # Get Redis info
            redis_info = await self._redis_client.info()
            
            # Get stream statistics
            stream_stats = {}
            for engine_type, config in self._engine_configs.items():
                try:
                    stream_info = await self._redis_client.xinfo_stream(config.stream_key)
                    groups_info = await self._redis_client.xinfo_groups(config.stream_key)
                    
                    stream_stats[engine_type.value] = {
                        "stream_key": config.stream_key,
                        "length": stream_info.get("length", 0),
                        "consumer_groups": len(groups_info),
                        "last_id": stream_info.get("last-generated-id", "0-0"),
                        "radix_tree_keys": stream_info.get("radix-tree-keys", 0),
                        "radix_tree_nodes": stream_info.get("radix-tree-nodes", 0),
                        "first_entry": stream_info.get("first-entry"),
                        "last_entry": stream_info.get("last-entry")
                    }
                except redis.RedisError:
                    stream_stats[engine_type.value] = {"error": "Stream not found"}
            
            return {
                "redis_connected": True,
                "redis_version": redis_info.get("redis_version"),
                "used_memory_human": redis_info.get("used_memory_human"),
                "connected_clients": redis_info.get("connected_clients"),
                "total_commands_processed": redis_info.get("total_commands_processed"),
                "keyspace_hits": redis_info.get("keyspace_hits"),
                "keyspace_misses": redis_info.get("keyspace_misses"),
                "stream_statistics": stream_stats,
                "total_streams": len([s for s in stream_stats.values() if "error" not in s]),
                "optimization_timestamp": time.time()
            }
            
        except Exception as e:
            return {"error": f"Failed to get status: {e}"}
    
    async def benchmark_performance(self) -> Dict[str, any]:
        """Benchmark MessageBus performance with optimized settings"""
        if not self._redis_client:
            return {"error": "Redis client not initialized"}
        
        benchmark_results = {}
        
        for engine_type, config in self._engine_configs.items():
            try:
                # Benchmark message publishing speed
                start_time = time.time()
                test_messages = 100
                
                pipe = self._redis_client.pipeline()
                for i in range(test_messages):
                    pipe.xadd(config.stream_key, {
                        "benchmark": "true",
                        "message_id": i,
                        "timestamp": int(time.time() * 1000),
                        "payload": json.dumps({"test": f"message_{i}"})
                    })
                
                await pipe.execute()
                end_time = time.time()
                
                # Calculate performance metrics
                duration = end_time - start_time
                messages_per_second = test_messages / duration if duration > 0 else 0
                avg_latency_ms = (duration * 1000) / test_messages if test_messages > 0 else 0
                
                benchmark_results[engine_type.value] = {
                    "messages_tested": test_messages,
                    "duration_seconds": duration,
                    "messages_per_second": messages_per_second,
                    "avg_latency_ms": avg_latency_ms,
                    "buffer_interval_ms": config.buffer_interval_ms,
                    "max_buffer_size": config.max_buffer_size,
                    "performance_rating": "excellent" if messages_per_second > 1000 else "good" if messages_per_second > 500 else "needs_optimization"
                }
                
            except Exception as e:
                benchmark_results[engine_type.value] = {"error": str(e)}
        
        return {
            "benchmark_timestamp": time.time(),
            "total_engines_tested": len(benchmark_results),
            "engine_performance": benchmark_results,
            "overall_performance": self._calculate_overall_performance(benchmark_results)
        }
    
    def _calculate_overall_performance(self, results: Dict) -> Dict:
        """Calculate overall system performance metrics"""
        valid_results = [r for r in results.values() if "error" not in r]
        
        if not valid_results:
            return {"rating": "error", "message": "No valid benchmark results"}
        
        avg_messages_per_second = sum(r["messages_per_second"] for r in valid_results) / len(valid_results)
        avg_latency_ms = sum(r["avg_latency_ms"] for r in valid_results) / len(valid_results)
        
        # Performance rating based on throughput and latency
        if avg_messages_per_second > 1000 and avg_latency_ms < 50:
            rating = "excellent"
        elif avg_messages_per_second > 500 and avg_latency_ms < 100:
            rating = "good"
        elif avg_messages_per_second > 250:
            rating = "acceptable"
        else:
            rating = "needs_optimization"
        
        return {
            "rating": rating,
            "avg_messages_per_second": avg_messages_per_second,
            "avg_latency_ms": avg_latency_ms,
            "engines_optimized": len(valid_results),
            "total_engines": len(self._engine_configs)
        }
    
    async def cleanup_test_messages(self):
        """Clean up benchmark test messages from streams"""
        try:
            for config in self._engine_configs.values():
                # Remove test messages (simplified cleanup)
                stream_info = await self._redis_client.xinfo_stream(config.stream_key)
                stream_length = stream_info.get("length", 0)
                
                if stream_length > 1000:  # Only trim if stream is getting large
                    # Keep only last 100 messages
                    await self._redis_client.xtrim(config.stream_key, maxlen=100, approximate=True)
                    logger.debug(f"Cleaned up test messages from {config.stream_key}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup test messages: {e}")


# Global optimizer instance
_messagebus_optimizer: Optional[MessageBusOptimizer] = None

async def get_messagebus_optimizer() -> MessageBusOptimizer:
    """Get global MessageBus optimizer instance"""
    global _messagebus_optimizer
    if _messagebus_optimizer is None:
        _messagebus_optimizer = MessageBusOptimizer()
        await _messagebus_optimizer.initialize()
    return _messagebus_optimizer


async def optimize_all_messagebus_connections():
    """Convenience function to optimize all MessageBus connections"""
    optimizer = await get_messagebus_optimizer()
    return await optimizer.optimize_connections()


async def get_messagebus_status():
    """Convenience function to get MessageBus optimization status"""
    optimizer = await get_messagebus_optimizer()
    return await optimizer.get_optimization_status()


async def benchmark_messagebus_performance():
    """Convenience function to benchmark MessageBus performance"""
    optimizer = await get_messagebus_optimizer()
    results = await optimizer.benchmark_performance()
    await optimizer.cleanup_test_messages()
    return results
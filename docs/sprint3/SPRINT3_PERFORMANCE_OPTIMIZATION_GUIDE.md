# Sprint 3 Performance Optimization Guide

## Overview

This comprehensive guide covers performance optimization strategies for Sprint 3 components including WebSocket infrastructure, real-time analytics, risk management systems, and database operations with specific benchmarks and monitoring techniques.

## Table of Contents

1. [Performance Benchmarks & Targets](#performance-benchmarks--targets)
2. [WebSocket Performance Optimization](#websocket-performance-optimization)
3. [Database & TimescaleDB Optimization](#database--timescaledb-optimization)
4. [Redis Performance Tuning](#redis-performance-tuning)
5. [Real-time Analytics Optimization](#real-time-analytics-optimization)
6. [Risk Calculation Performance](#risk-calculation-performance)
7. [Frontend Performance Optimization](#frontend-performance-optimization)
8. [Caching Strategies](#caching-strategies)
9. [Load Testing & Benchmarking](#load-testing--benchmarking)
10. [Production Monitoring](#production-monitoring)

---

## Performance Benchmarks & Targets

### Target Performance Metrics

| Component | Metric | Target | Critical Threshold |
|-----------|--------|--------|--------------------|
| API Response Time | Average | <200ms | >1000ms |
| WebSocket Latency | End-to-end | <50ms | >200ms |
| WebSocket Throughput | Messages/sec | >10,000 | <1,000 |
| Database Query Time | Complex queries | <500ms | >2000ms |
| Risk Calculation | VaR calculation | <1000ms | >5000ms |
| Analytics Update | Real-time metrics | <100ms | >500ms |
| Memory Usage | Per service | <2GB | >4GB |
| CPU Usage | Sustained | <70% | >90% |

### Performance Testing Framework

```python
# performance_tests/benchmarks.py
import asyncio
import time
import statistics
from typing import List, Dict, Any
import aiohttp
import websockets
import json

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http", "ws")
        self.results = {}
    
    async def run_api_benchmark(self, endpoint: str, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark API endpoint performance."""
        response_times = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.text()
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    response_times.append(response_time)
                except Exception as e:
                    print(f"Request {i} failed: {e}")
        
        return {
            'mean': statistics.mean(response_times),
            'median': statistics.median(response_times),
            'p95': self.percentile(response_times, 95),
            'p99': self.percentile(response_times, 99),
            'min': min(response_times),
            'max': max(response_times),
            'requests_per_second': iterations / (max(response_times) - min(response_times)) * 1000
        }
    
    async def run_websocket_benchmark(self, duration: int = 60) -> Dict[str, Any]:
        """Benchmark WebSocket performance."""
        messages_sent = 0
        messages_received = 0
        latencies = []
        
        async def websocket_client():
            nonlocal messages_sent, messages_received, latencies
            
            uri = f"{self.ws_url}/ws/system/health"
            async with websockets.connect(uri) as websocket:
                # Start sending messages
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    # Send message with timestamp
                    message = {
                        "type": "ping",
                        "timestamp": time.time() * 1000
                    }
                    
                    await websocket.send(json.dumps(message))
                    messages_sent += 1
                    
                    # Receive response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        received_time = time.time() * 1000
                        
                        response_data = json.loads(response)
                        if "timestamp" in response_data:
                            latency = received_time - response_data["timestamp"]
                            latencies.append(latency)
                        
                        messages_received += 1
                    except asyncio.TimeoutError:
                        pass
                    
                    await asyncio.sleep(0.01)  # 10ms interval
        
        await websocket_client()
        
        return {
            'messages_sent': messages_sent,
            'messages_received': messages_received,
            'message_loss_rate': (messages_sent - messages_received) / messages_sent,
            'throughput_sent': messages_sent / duration,
            'throughput_received': messages_received / duration,
            'latency': {
                'mean': statistics.mean(latencies) if latencies else 0,
                'p95': self.percentile(latencies, 95) if latencies else 0,
                'p99': self.percentile(latencies, 99) if latencies else 0
            }
        }
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c

# Usage example
async def run_comprehensive_benchmark():
    benchmark = PerformanceBenchmark()
    
    # API benchmarks
    health_results = await benchmark.run_api_benchmark("/health")
    analytics_results = await benchmark.run_api_benchmark("/api/v1/sprint3/analytics/portfolio/PORTFOLIO_001/summary")
    
    # WebSocket benchmark
    ws_results = await benchmark.run_websocket_benchmark(duration=30)
    
    print("API Health Endpoint:", health_results)
    print("Analytics Endpoint:", analytics_results)
    print("WebSocket Performance:", ws_results)
```

---

## WebSocket Performance Optimization

### Connection Pool Optimization

```python
# backend/websocket/optimized_manager.py
import asyncio
import weakref
from collections import defaultdict
from typing import Dict, Set, Optional
import uvloop

class OptimizedWebSocketManager:
    """High-performance WebSocket connection manager."""
    
    def __init__(self, max_connections: int = 10000):
        self.max_connections = max_connections
        self.connections: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.message_queue = asyncio.Queue(maxsize=100000)
        self.broadcasting = False
        
        # Use uvloop for better performance
        if not isinstance(asyncio.get_event_loop(), uvloop.Loop):
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    async def register_connection(
        self, 
        connection_id: str, 
        websocket, 
        connection_metadata: Dict[str, str]
    ) -> bool:
        """Register WebSocket connection with optimization."""
        if len(self.connections) >= self.max_connections:
            return False
        
        # Add to weak reference set to prevent memory leaks
        self.connections[connection_metadata.get('user_id', 'anonymous')].add(websocket)
        
        # Start broadcasting if not already started
        if not self.broadcasting:
            asyncio.create_task(self.broadcast_worker())
            self.broadcasting = True
        
        return True
    
    async def broadcast_worker(self):
        """Optimized message broadcasting worker."""
        batch_size = 1000
        batch_timeout = 0.01  # 10ms batch timeout
        
        while True:
            messages = []
            
            # Collect messages in batches
            try:
                # Get first message (blocking)
                message = await self.message_queue.get()
                messages.append(message)
                
                # Collect additional messages (non-blocking)
                start_time = asyncio.get_event_loop().time()
                while (len(messages) < batch_size and 
                       asyncio.get_event_loop().time() - start_time < batch_timeout):
                    try:
                        message = self.message_queue.get_nowait()
                        messages.append(message)
                    except asyncio.QueueEmpty:
                        break
            except asyncio.CancelledError:
                break
            
            # Broadcast batch
            if messages:
                await self.broadcast_batch(messages)
    
    async def broadcast_batch(self, messages: List[Dict]) -> None:
        """Broadcast messages in batch for efficiency."""
        # Group messages by topic for efficient filtering
        topic_messages = defaultdict(list)
        for message in messages:
            topic_messages[message.get('topic', 'default')].append(message)
        
        # Send to relevant connections
        for topic, topic_msgs in topic_messages.items():
            await self.send_to_topic_subscribers(topic, topic_msgs)
    
    async def send_to_topic_subscribers(self, topic: str, messages: List[Dict]) -> None:
        """Send messages to topic subscribers efficiently."""
        if topic not in self.subscriptions:
            return
        
        # Serialize messages once
        serialized_messages = [json.dumps(msg) for msg in messages]
        
        # Send to all subscribers
        tasks = []
        for connection_id in self.subscriptions[topic]:
            websocket = self.get_connection(connection_id)
            if websocket:
                # Create task for concurrent sending
                task = asyncio.create_task(
                    self.safe_send_batch(websocket, serialized_messages)
                )
                tasks.append(task)
        
        # Wait for all sends to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def safe_send_batch(self, websocket, messages: List[str]) -> None:
        """Safely send batch of messages to WebSocket."""
        try:
            for message in messages:
                await websocket.send(message)
        except Exception as e:
            # Remove failed connection
            await self.remove_connection(websocket)
```

### Message Compression

```python
# backend/websocket/compression.py
import gzip
import json
from typing import Dict, Any

class MessageCompressor:
    """WebSocket message compression for large payloads."""
    
    def __init__(self, compression_threshold: int = 1024):
        self.compression_threshold = compression_threshold
    
    def compress_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Compress large messages."""
        serialized = json.dumps(message)
        
        if len(serialized) > self.compression_threshold:
            compressed = gzip.compress(serialized.encode('utf-8'))
            return {
                'type': 'compressed',
                'data': compressed.hex(),
                'original_size': len(serialized),
                'compressed_size': len(compressed)
            }
        
        return message
    
    def decompress_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress received messages."""
        if message.get('type') == 'compressed':
            compressed_data = bytes.fromhex(message['data'])
            decompressed = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(decompressed)
        
        return message
```

### WebSocket Load Balancing

```python
# backend/websocket/load_balancer.py
import random
from typing import List, Dict
import asyncio

class WebSocketLoadBalancer:
    """Load balancer for WebSocket connections."""
    
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.server_stats = {server: {'connections': 0, 'load': 0.0} for server in servers}
    
    def get_optimal_server(self) -> str:
        """Get server with lowest load."""
        return min(self.server_stats.keys(), 
                  key=lambda s: self.server_stats[s]['load'])
    
    async def redistribute_connections(self) -> None:
        """Redistribute connections for optimal load balancing."""
        # Calculate average load
        total_connections = sum(stats['connections'] for stats in self.server_stats.values())
        target_per_server = total_connections // len(self.servers)
        
        # Identify overloaded servers
        for server, stats in self.server_stats.items():
            if stats['connections'] > target_per_server * 1.2:  # 20% over target
                excess = stats['connections'] - target_per_server
                await self.migrate_connections(server, excess)
    
    async def migrate_connections(self, from_server: str, count: int) -> None:
        """Migrate connections to less loaded servers."""
        target_server = self.get_optimal_server()
        if target_server == from_server:
            return
        
        # Implementation would migrate actual connections
        # This is a simplified example
        self.server_stats[from_server]['connections'] -= count
        self.server_stats[target_server]['connections'] += count
```

---

## Database & TimescaleDB Optimization

### Advanced Indexing Strategies

```sql
-- Performance-optimized indexes for Sprint 3 tables

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_analytics_portfolio_time_metrics 
ON analytics_metrics (portfolio_id, timestamp DESC, metric_type) 
INCLUDE (value, metadata);

-- Partial indexes for active data
CREATE INDEX CONCURRENTLY idx_risk_limits_active_portfolio 
ON risk_limits (portfolio_id, limit_type) 
WHERE active = true AND deleted_at IS NULL;

-- GiST index for complex queries
CREATE INDEX CONCURRENTLY idx_websocket_metadata_gin 
ON websocket_connections USING GIN (metadata);

-- Hash index for exact matches
CREATE INDEX CONCURRENTLY idx_portfolio_id_hash 
ON portfolio_data USING HASH (portfolio_id);

-- Expression indexes for computed values
CREATE INDEX CONCURRENTLY idx_risk_utilization 
ON risk_limits ((current_value / threshold_value)) 
WHERE active = true;

-- Covering indexes to avoid table lookups
CREATE INDEX CONCURRENTLY idx_trades_portfolio_covering 
ON trades (portfolio_id, timestamp DESC) 
INCLUDE (symbol, quantity, price, trade_type);
```

### TimescaleDB Hypertable Optimization

```sql
-- Optimize chunk intervals based on data volume
SELECT set_chunk_time_interval('analytics_metrics', INTERVAL '1 hour');
SELECT set_chunk_time_interval('risk_events', INTERVAL '15 minutes');
SELECT set_chunk_time_interval('websocket_metrics', INTERVAL '5 minutes');

-- Add compression policies for older data
SELECT add_compression_policy('analytics_metrics', INTERVAL '1 day');
SELECT add_compression_policy('risk_events', INTERVAL '6 hours');
SELECT add_compression_policy('websocket_metrics', INTERVAL '2 hours');

-- Retention policies for automatic cleanup
SELECT add_retention_policy('websocket_metrics', INTERVAL '7 days');
SELECT add_retention_policy('debug_logs', INTERVAL '3 days');

-- Optimize compression settings
ALTER TABLE analytics_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'portfolio_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW analytics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    portfolio_id,
    metric_type,
    avg(value) as avg_value,
    max(value) as max_value,
    min(value) as min_value,
    count(*) as count
FROM analytics_metrics
GROUP BY bucket, portfolio_id, metric_type;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('analytics_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');
```

### Query Optimization

```python
# backend/database/optimized_queries.py
from sqlalchemy import text, select
from sqlalchemy.orm import selectinload, joinedload
import asyncio

class OptimizedQueries:
    """Optimized database queries for Sprint 3."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def get_portfolio_analytics_bulk(
        self, 
        portfolio_ids: List[str], 
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Optimized bulk analytics query."""
        
        # Use raw SQL for complex aggregations
        query = text("""
            SELECT 
                portfolio_id,
                metric_type,
                time_bucket('5 minutes', timestamp) as bucket,
                avg(value) as avg_value,
                first(value, timestamp) as first_value,
                last(value, timestamp) as last_value,
                count(*) as data_points
            FROM analytics_metrics 
            WHERE portfolio_id = ANY(:portfolio_ids)
                AND timestamp BETWEEN :start_date AND :end_date
            GROUP BY portfolio_id, metric_type, bucket
            ORDER BY portfolio_id, metric_type, bucket DESC
        """)
        
        result = await self.db.execute(
            query,
            {
                'portfolio_ids': portfolio_ids,
                'start_date': start_date,
                'end_date': end_date
            }
        )
        
        return result.fetchall()
    
    async def get_risk_metrics_with_limits(
        self, 
        portfolio_id: str
    ) -> Dict[str, Any]:
        """Optimized query combining risk metrics and limits."""
        
        # Use JOIN to get data in single query
        query = text("""
            WITH latest_metrics AS (
                SELECT DISTINCT ON (metric_type)
                    metric_type,
                    value,
                    timestamp
                FROM risk_metrics
                WHERE portfolio_id = :portfolio_id
                    AND timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY metric_type, timestamp DESC
            ),
            active_limits AS (
                SELECT 
                    limit_type,
                    threshold_value,
                    warning_threshold,
                    current_value,
                    (current_value / threshold_value) as utilization
                FROM risk_limits
                WHERE portfolio_id = :portfolio_id
                    AND active = true
                    AND deleted_at IS NULL
            )
            SELECT 
                m.metric_type,
                m.value as current_value,
                m.timestamp as last_updated,
                l.threshold_value,
                l.warning_threshold,
                l.utilization,
                CASE 
                    WHEN l.utilization > 1.0 THEN 'breach'
                    WHEN l.utilization > 0.8 THEN 'warning'
                    ELSE 'normal'
                END as status
            FROM latest_metrics m
            LEFT JOIN active_limits l ON m.metric_type = l.limit_type
            ORDER BY l.utilization DESC NULLS LAST
        """)
        
        result = await self.db.execute(
            query,
            {'portfolio_id': portfolio_id}
        )
        
        return result.fetchall()
    
    async def batch_insert_metrics(
        self, 
        metrics_data: List[Dict[str, Any]]
    ) -> None:
        """Optimized batch insert for metrics."""
        
        if not metrics_data:
            return
        
        # Use COPY for fastest bulk insert
        query = text("""
            INSERT INTO analytics_metrics 
            (portfolio_id, metric_type, value, timestamp, metadata)
            VALUES 
        """ + ",".join([
            "(:portfolio_id_{i}, :metric_type_{i}, :value_{i}, :timestamp_{i}, :metadata_{i})".format(i=i)
            for i in range(len(metrics_data))
        ]))
        
        # Prepare parameters
        params = {}
        for i, data in enumerate(metrics_data):
            params[f'portfolio_id_{i}'] = data['portfolio_id']
            params[f'metric_type_{i}'] = data['metric_type']
            params[f'value_{i}'] = data['value']
            params[f'timestamp_{i}'] = data['timestamp']
            params[f'metadata_{i}'] = data.get('metadata', {})
        
        await self.db.execute(query, params)
        await self.db.commit()
```

### Connection Pool Tuning

```python
# backend/database/connection_pool.py
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool, QueuePool
import asyncio

def create_optimized_engine(database_url: str, **kwargs):
    """Create optimized database engine."""
    
    # Production settings
    return create_async_engine(
        database_url,
        
        # Connection pool settings
        poolclass=QueuePool,
        pool_size=30,           # Base connections
        max_overflow=70,        # Additional connections
        pool_timeout=30,        # Wait timeout
        pool_recycle=3600,      # Recycle after 1 hour
        pool_pre_ping=True,     # Validate connections
        
        # Query settings
        echo=False,             # Disable SQL logging in production
        future=True,            # Use SQLAlchemy 2.0 style
        
        # Connection arguments
        connect_args={
            "server_settings": {
                "application_name": "nautilus_sprint3",
                "tcp_keepalives_idle": "600",
                "tcp_keepalives_interval": "30",
                "tcp_keepalives_count": "3",
            },
            "command_timeout": 60,
        },
        
        **kwargs
    )

# Connection health monitoring
class ConnectionPoolMonitor:
    """Monitor database connection pool health."""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status."""
        pool = self.engine.pool
        
        return {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid(),
        }
    
    async def monitor_pool_health(self):
        """Continuous pool health monitoring."""
        while True:
            try:
                status = await self.get_pool_status()
                
                # Alert if pool utilization is high
                utilization = status['checked_out'] / (status['size'] + status['overflow'])
                if utilization > 0.8:
                    logger.warning(f"High pool utilization: {utilization:.2%}")
                
                # Alert if many invalid connections
                if status['invalid'] > 5:
                    logger.error(f"Many invalid connections: {status['invalid']}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring connection pool: {e}")
                await asyncio.sleep(60)
```

---

## Redis Performance Tuning

### Redis Configuration Optimization

```bash
# redis.conf optimizations for Sprint 3

# Memory management
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence settings for performance
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error no

# Network and connection settings
tcp-keepalive 300
timeout 300
tcp-backlog 511
maxclients 65000

# Performance tuning
hz 10
dynamic-hz yes

# Disable slow operations in production
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Enable pipelining
tcp-nodelay yes

# Memory usage optimization
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
```

### Redis Client Optimization

```python
# backend/redis/optimized_client.py
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
import asyncio
import json
from typing import List, Dict, Any
import pickle

class OptimizedRedisClient:
    """High-performance Redis client for Sprint 3."""
    
    def __init__(self, redis_url: str):
        # Create optimized connection pool
        self.pool = ConnectionPool.from_url(
            redis_url,
            max_connections=100,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Pipelining for batch operations
        self.pipeline_batch_size = 100
    
    async def batch_set(self, key_value_pairs: Dict[str, Any], ttl: int = None) -> None:
        """Optimized batch set operation."""
        pipeline = self.client.pipeline()
        
        for key, value in key_value_pairs.items():
            # Use efficient serialization
            serialized_value = self.serialize_value(value)
            
            if ttl:
                pipeline.setex(key, ttl, serialized_value)
            else:
                pipeline.set(key, serialized_value)
        
        await pipeline.execute()
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Optimized batch get operation."""
        pipeline = self.client.pipeline()
        
        for key in keys:
            pipeline.get(key)
        
        results = await pipeline.execute()
        
        return {
            key: self.deserialize_value(value) if value else None
            for key, value in zip(keys, results)
        }
    
    def serialize_value(self, value: Any) -> bytes:
        """Efficient value serialization."""
        if isinstance(value, (dict, list)):
            # Use pickle for complex objects (faster than JSON)
            return pickle.dumps(value)
        elif isinstance(value, str):
            return value.encode('utf-8')
        else:
            return str(value).encode('utf-8')
    
    def deserialize_value(self, value: bytes) -> Any:
        """Efficient value deserialization."""
        try:
            # Try pickle first
            return pickle.loads(value)
        except:
            try:
                # Fallback to JSON
                return json.loads(value.decode('utf-8'))
            except:
                # Return as string
                return value.decode('utf-8')
    
    async def publish_batch(self, channel_messages: Dict[str, Any]) -> None:
        """Optimized batch publishing."""
        pipeline = self.client.pipeline()
        
        for channel, message in channel_messages.items():
            serialized_message = json.dumps(message)
            pipeline.publish(channel, serialized_message)
        
        await pipeline.execute()
    
    async def cached_computation(
        self, 
        cache_key: str, 
        computation_func,
        ttl: int = 300,
        **kwargs
    ) -> Any:
        """Cache computation results with optimization."""
        # Try to get from cache first
        cached_result = await self.client.get(cache_key)
        if cached_result:
            return self.deserialize_value(cached_result)
        
        # Compute result
        result = await computation_func(**kwargs)
        
        # Cache result asynchronously
        asyncio.create_task(
            self.client.setex(cache_key, ttl, self.serialize_value(result))
        )
        
        return result

# Memory usage monitoring
class RedisMemoryMonitor:
    """Monitor Redis memory usage and optimization."""
    
    def __init__(self, redis_client):
        self.client = redis_client
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        info = await self.client.info('memory')
        
        return {
            'used_memory': info['used_memory'],
            'used_memory_human': info['used_memory_human'],
            'used_memory_peak': info['used_memory_peak'],
            'used_memory_peak_human': info['used_memory_peak_human'],
            'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 1.0),
            'memory_efficiency': info['used_memory'] / info['used_memory_rss'] if info.get('used_memory_rss') else 1.0
        }
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization tasks."""
        results = {}
        
        # Get current stats
        before_stats = await self.get_memory_stats()
        results['before'] = before_stats
        
        # Run memory optimization
        try:
            # Defragment memory if fragmentation ratio is high
            if before_stats['memory_fragmentation_ratio'] > 1.5:
                await self.client.execute_command('MEMORY', 'PURGE')
                results['defragmentation'] = 'executed'
            
            # Cleanup expired keys
            await self.client.execute_command('MEMORY', 'PURGE')
            
            # Get after stats
            after_stats = await self.get_memory_stats()
            results['after'] = after_stats
            results['memory_saved'] = before_stats['used_memory'] - after_stats['used_memory']
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
```

---

## Real-time Analytics Optimization

### Streaming Analytics Engine

```python
# backend/analytics/streaming_engine.py
import asyncio
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Any, Callable
import time

class StreamingAnalyticsEngine:
    """High-performance streaming analytics engine."""
    
    def __init__(self, max_window_size: int = 10000):
        self.data_windows = {}  # Sliding windows for each portfolio
        self.calculators = {}   # Pre-compiled calculation functions
        self.results_cache = {} # Recent calculation results
        self.max_window_size = max_window_size
        
        # Performance counters
        self.calculations_per_second = 0
        self.last_performance_check = time.time()
    
    def register_calculator(self, name: str, func: Callable) -> None:
        """Register optimized calculation function."""
        self.calculators[name] = func
    
    async def process_data_point(
        self, 
        portfolio_id: str, 
        data_point: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single data point efficiently."""
        # Add to sliding window
        if portfolio_id not in self.data_windows:
            self.data_windows[portfolio_id] = deque(maxlen=self.max_window_size)
        
        self.data_windows[portfolio_id].append(data_point)
        
        # Run incremental calculations
        results = await self.calculate_incremental_metrics(portfolio_id, data_point)
        
        # Update performance counter
        self.calculations_per_second += 1
        await self.check_performance()
        
        return results
    
    async def calculate_incremental_metrics(
        self, 
        portfolio_id: str, 
        new_data_point: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics incrementally for performance."""
        window = self.data_windows[portfolio_id]
        results = {}
        
        if len(window) < 2:
            return results
        
        # Convert to numpy array for fast calculations
        values = np.array([point.get('value', 0) for point in window])
        timestamps = np.array([point.get('timestamp') for point in window])
        
        # Fast vectorized calculations
        results.update({
            'current_value': float(values[-1]),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'trend': self.calculate_trend(values),
            'volatility': self.calculate_rolling_volatility(values),
            'momentum': self.calculate_momentum(values),
        })
        
        # Run custom calculators
        for name, calculator in self.calculators.items():
            try:
                result = await calculator(window, new_data_point)
                results[name] = result
            except Exception as e:
                logger.warning(f"Calculator {name} failed: {e}")
        
        # Cache results
        self.results_cache[portfolio_id] = results
        
        return results
    
    @staticmethod
    def calculate_trend(values: np.ndarray) -> float:
        """Calculate trend using linear regression slope."""
        if len(values) < 10:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    @staticmethod
    def calculate_rolling_volatility(values: np.ndarray, window: int = 20) -> float:
        """Calculate rolling volatility efficiently."""
        if len(values) < window:
            return float(np.std(values))
        
        returns = np.diff(values[-window:]) / values[-window:-1]
        return float(np.std(returns) * np.sqrt(252))  # Annualized
    
    @staticmethod
    def calculate_momentum(values: np.ndarray, periods: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """Calculate momentum over multiple periods."""
        momentum = {}
        
        for period in periods:
            if len(values) > period:
                momentum[f'momentum_{period}'] = float(
                    (values[-1] - values[-(period+1)]) / values[-(period+1)]
                )
        
        return momentum
    
    async def check_performance(self) -> None:
        """Monitor performance and adjust if needed."""
        current_time = time.time()
        time_diff = current_time - self.last_performance_check
        
        if time_diff >= 1.0:  # Check every second
            cps = self.calculations_per_second / time_diff
            
            if cps < 1000:  # Less than 1000 calculations per second
                logger.warning(f"Low calculation performance: {cps:.1f} calc/sec")
            
            # Reset counters
            self.calculations_per_second = 0
            self.last_performance_check = current_time

# High-performance risk calculations
class OptimizedRiskCalculator:
    """Optimized risk calculation engine."""
    
    def __init__(self):
        self.var_cache = {}
        self.correlation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def calculate_portfolio_var(
        self, 
        portfolio_data: Dict[str, np.ndarray],
        confidence_level: float = 0.95,
        method: str = 'parametric'
    ) -> float:
        """Optimized VaR calculation."""
        cache_key = f"var_{hash(str(portfolio_data))}_{confidence_level}_{method}"
        
        # Check cache
        if cache_key in self.var_cache:
            cached_time, cached_value = self.var_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_value
        
        # Calculate VaR based on method
        if method == 'parametric':
            var_value = await self.parametric_var(portfolio_data, confidence_level)
        elif method == 'historical':
            var_value = await self.historical_var(portfolio_data, confidence_level)
        else:
            var_value = await self.monte_carlo_var(portfolio_data, confidence_level)
        
        # Cache result
        self.var_cache[cache_key] = (time.time(), var_value)
        return var_value
    
    async def parametric_var(
        self, 
        portfolio_data: Dict[str, np.ndarray], 
        confidence_level: float
    ) -> float:
        """Fast parametric VaR calculation."""
        # Extract portfolio values and weights
        values = list(portfolio_data.values())
        weights = np.array([len(v) for v in values])  # Simplified weighting
        weights = weights / np.sum(weights)
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros(len(values[0]))
        for i, asset_values in enumerate(values):
            returns = np.diff(asset_values) / asset_values[:-1]
            portfolio_returns += weights[i] * returns
        
        # Calculate VaR
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        z_score = norm.ppf(1 - confidence_level)
        
        var_value = mean_return + z_score * std_return
        return float(var_value)
```

This performance optimization guide provides comprehensive strategies for maximizing Sprint 3 system performance across all components. The optimizations include specific code implementations, configuration settings, and monitoring techniques for production environments.
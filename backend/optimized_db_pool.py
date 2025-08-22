"""
Optimized Database Connection Pooling
=====================================

High-performance PostgreSQL connection pooling with intelligent management.
"""

import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class PoolStrategy(str, Enum):
    """Connection pool strategies for different workloads."""
    HIGH_THROUGHPUT = "high_throughput"     # Many concurrent connections
    BALANCED = "balanced"                   # Balanced performance
    CONSERVATIVE = "conservative"           # Few connections, low overhead


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_connections: int
    max_connections: int
    max_queries: int
    max_inactive_connection_lifetime: float
    command_timeout: Optional[float] = None
    server_settings: Optional[Dict[str, str]] = None


class OptimizedDBPool:
    """Optimized PostgreSQL connection pool with intelligent management."""
    
    def __init__(self, database_url: str, strategy: PoolStrategy = PoolStrategy.BALANCED):
        self.database_url = database_url
        self.strategy = strategy
        self.pool: Optional[asyncpg.Pool] = None
        
        # Pool configurations by strategy
        self.pool_configs = {
            PoolStrategy.HIGH_THROUGHPUT: PoolConfig(
                min_connections=10,
                max_connections=50,
                max_queries=10000,
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                command_timeout=30.0,
                server_settings={
                    "application_name": "nautilus_high_throughput",
                    "tcp_keepalives_idle": "600",
                    "tcp_keepalives_interval": "30",
                    "tcp_keepalives_count": "3"
                }
            ),
            PoolStrategy.BALANCED: PoolConfig(
                min_connections=5,
                max_connections=20,
                max_queries=5000,
                max_inactive_connection_lifetime=600.0,  # 10 minutes
                command_timeout=60.0,
                server_settings={
                    "application_name": "nautilus_balanced",
                    "tcp_keepalives_idle": "900",
                    "tcp_keepalives_interval": "60",
                    "tcp_keepalives_count": "2"
                }
            ),
            PoolStrategy.CONSERVATIVE: PoolConfig(
                min_connections=2,
                max_connections=10,
                max_queries=2000,
                max_inactive_connection_lifetime=1800.0,  # 30 minutes
                command_timeout=120.0,
                server_settings={
                    "application_name": "nautilus_conservative"
                }
            )
        }
        
        # Performance metrics
        self.total_connections_created = 0
        self.total_queries_executed = 0
        self.total_query_time = 0.0
        self.failed_connections = 0
        self.failed_queries = 0
        
        # Health monitoring
        self.last_health_check = None
        self.health_check_interval = 60.0  # 1 minute
    
    async def initialize(self) -> bool:
        """Initialize the connection pool."""
        try:
            config = self.pool_configs[self.strategy]
            
            logger.info(f"Initializing database pool with strategy: {self.strategy.value}")
            logger.info(f"Pool config: min={config.min_connections}, max={config.max_connections}")
            
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=config.min_connections,
                max_size=config.max_connections,
                max_queries=config.max_queries,
                max_inactive_connection_lifetime=config.max_inactive_connection_lifetime,
                command_timeout=config.command_timeout,
                server_settings=config.server_settings or {}
            )
            
            # Test the pool with a simple query
            async with self.pool.acquire() as connection:
                await connection.fetchval("SELECT 1")
            
            logger.info("✅ Database pool initialized successfully")
            return True
            
        except Exception as e:
            self.failed_connections += 1
            logger.error(f"❌ Failed to initialize database pool: {e}")
            return False
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a connection from the pool with proper error handling."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        connection = None
        start_time = asyncio.get_event_loop().time()
        
        try:
            connection = await self.pool.acquire()
            yield connection
            
        except Exception as e:
            self.failed_queries += 1
            logger.error(f"Database operation failed: {e}")
            raise
            
        finally:
            if connection:
                try:
                    await self.pool.release(connection)
                except Exception as e:
                    logger.error(f"Failed to release connection: {e}")
            
            # Track performance metrics
            query_time = asyncio.get_event_loop().time() - start_time
            self.total_queries_executed += 1
            self.total_query_time += query_time
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query with connection pooling."""
        async with self.acquire_connection() as connection:
            return await connection.fetchval(query, *args)
    
    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        """Execute multiple queries efficiently."""
        async with self.acquire_connection() as connection:
            return await connection.executemany(query, args_list)
    
    async def fetch_rows(self, query: str, *args) -> List[Dict]:
        """Fetch multiple rows as dictionaries."""
        async with self.acquire_connection() as connection:
            rows = await connection.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch one row as dictionary."""
        async with self.acquire_connection() as connection:
            row = await connection.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def transaction(self):
        """Get a transaction context manager."""
        return self.pool.acquire()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the database pool."""
        if not self.pool:
            return {
                "status": "unhealthy",
                "error": "Pool not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test connection with simple query
            async with self.acquire_connection() as connection:
                await connection.fetchval("SELECT 1")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get pool statistics
            pool_stats = {
                "size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "idle_connections": self.pool.get_idle_size(),
            }
            
            self.last_health_check = datetime.now()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "pool_stats": pool_stats,
                "strategy": self.strategy.value,
                "timestamp": self.last_health_check.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        if not self.pool:
            return {"error": "Pool not initialized"}
        
        avg_query_time = (
            self.total_query_time / self.total_queries_executed 
            if self.total_queries_executed > 0 else 0
        )
        
        success_rate = (
            (self.total_queries_executed - self.failed_queries) / self.total_queries_executed * 100
            if self.total_queries_executed > 0 else 0
        )
        
        return {
            "strategy": self.strategy.value,
            "pool_stats": {
                "current_size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "idle_connections": self.pool.get_idle_size(),
            },
            "performance_metrics": {
                "total_queries_executed": self.total_queries_executed,
                "total_query_time_seconds": round(self.total_query_time, 3),
                "average_query_time_ms": round(avg_query_time * 1000, 2),
                "failed_queries": self.failed_queries,
                "success_rate_percentage": round(success_rate, 2),
                "total_connections_created": self.total_connections_created,
                "failed_connections": self.failed_connections
            },
            "configuration": {
                "min_connections": self.pool_configs[self.strategy].min_connections,
                "max_connections": self.pool_configs[self.strategy].max_connections,
                "max_queries": self.pool_configs[self.strategy].max_queries,
                "command_timeout": self.pool_configs[self.strategy].command_timeout,
                "max_inactive_lifetime": self.pool_configs[self.strategy].max_inactive_connection_lifetime
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def optimize_for_workload(self, workload_type: str) -> bool:
        """Dynamically optimize pool for specific workload."""
        strategy_mapping = {
            "high_frequency_trading": PoolStrategy.HIGH_THROUGHPUT,
            "analytics": PoolStrategy.BALANCED,
            "reporting": PoolStrategy.CONSERVATIVE,
            "backfill": PoolStrategy.HIGH_THROUGHPUT
        }
        
        new_strategy = strategy_mapping.get(workload_type)
        if new_strategy and new_strategy != self.strategy:
            logger.info(f"Optimizing database pool for workload: {workload_type}")
            
            # Close current pool
            if self.pool:
                await self.pool.close()
            
            # Initialize with new strategy
            self.strategy = new_strategy
            return await self.initialize()
        
        return True


# Global optimized database pool instance
optimized_db_pool = OptimizedDBPool(
    database_url="postgresql://nautilus:nautilus123@localhost:5432/nautilus",
    strategy=PoolStrategy.BALANCED
)
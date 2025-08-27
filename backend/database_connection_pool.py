#!/usr/bin/env python3
"""
Standardized Database Connection Pool for All Engines
Provides optimized PostgreSQL connection pooling for the Nautilus platform.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """
    Centralized database connection pool manager for all Nautilus engines.
    Provides optimized connection pooling with engine-specific configurations.
    """
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.pool: Optional[Pool] = None
        self._initialized = False
        
        # Engine-specific connection pool configurations
        self.pool_configs = {
            "analytics": {"min_size": 5, "max_size": 15, "command_timeout": 30},
            "risk": {"min_size": 3, "max_size": 10, "command_timeout": 20},
            "factor": {"min_size": 8, "max_size": 20, "command_timeout": 45},
            "ml": {"min_size": 4, "max_size": 12, "command_timeout": 60},
            "features": {"min_size": 3, "max_size": 8, "command_timeout": 25},
            "websocket": {"min_size": 2, "max_size": 6, "command_timeout": 15},
            "strategy": {"min_size": 3, "max_size": 9, "command_timeout": 30},
            "marketdata": {"min_size": 6, "max_size": 18, "command_timeout": 20},
            "portfolio": {"min_size": 5, "max_size": 15, "command_timeout": 35},
            "collateral": {"min_size": 4, "max_size": 12, "command_timeout": 25},
            "vpin": {"min_size": 3, "max_size": 10, "command_timeout": 40},
            "backtesting": {"min_size": 2, "max_size": 8, "command_timeout": 120},
            "toraniko": {"min_size": 3, "max_size": 9, "command_timeout": 30},
        }
        
        # Default configuration for unknown engines
        self.default_config = {"min_size": 2, "max_size": 8, "command_timeout": 30}
        
    async def initialize(self) -> bool:
        """Initialize the connection pool for this engine"""
        if self._initialized:
            return True
            
        try:
            # Get database URL from environment
            database_url = os.getenv(
                "DATABASE_URL", 
                "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
            )
            
            # Get engine-specific configuration
            config = self.pool_configs.get(self.engine_name, self.default_config)
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=config["min_size"],
                max_size=config["max_size"],
                command_timeout=config["command_timeout"],
                server_settings={
                    'application_name': f'nautilus-{self.engine_name}',
                    'jit': 'off',  # Disable JIT for consistent performance
                }
            )
            
            # Test the connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("SELECT 1 as test, current_timestamp as time")
                logger.info(f"✅ {self.engine_name} engine database pool initialized: {result}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pool for {self.engine_name}: {e}")
            return False
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query using the connection pool"""
        if not self._initialized:
            await self.initialize()
            
        if not self.pool:
            raise RuntimeError(f"Database pool not initialized for {self.engine_name}")
            
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_single(self, query: str, *args) -> Any:
        """Execute a query that returns a single row"""
        if not self._initialized:
            await self.initialize()
            
        if not self.pool:
            raise RuntimeError(f"Database pool not initialized for {self.engine_name}")
            
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a command that doesn't return data"""
        if not self._initialized:
            await self.initialize()
            
        if not self.pool:
            raise RuntimeError(f"Database pool not initialized for {self.engine_name}")
            
        async with self.pool.acquire() as conn:
            return await conn.execute(command, *args)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self.pool:
            return {"status": "not_initialized"}
            
        return {
            "engine": self.engine_name,
            "pool_size": self.pool.get_size(),
            "pool_min": self.pool.get_min_size(),
            "pool_max": self.pool.get_max_size(),
            "idle_connections": self.pool.get_idle_size(),
            "active_connections": self.pool.get_size() - self.pool.get_idle_size(),
            "initialized": self._initialized,
        }
    
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info(f"🔌 {self.engine_name} database pool closed")


# Global pool instances for each engine
_engine_pools: Dict[str, DatabaseConnectionPool] = {}

async def get_database_pool(engine_name: str) -> DatabaseConnectionPool:
    """Get or create a database connection pool for an engine"""
    if engine_name not in _engine_pools:
        _engine_pools[engine_name] = DatabaseConnectionPool(engine_name)
        await _engine_pools[engine_name].initialize()
    
    return _engine_pools[engine_name]

async def close_all_pools():
    """Close all database connection pools"""
    for pool in _engine_pools.values():
        await pool.close()
    _engine_pools.clear()

# Convenience functions for engines
async def db_query(engine_name: str, query: str, *args) -> Any:
    """Execute a database query for an engine"""
    pool = await get_database_pool(engine_name)
    return await pool.execute_query(query, *args)

async def db_single(engine_name: str, query: str, *args) -> Any:
    """Execute a database query that returns a single row"""
    pool = await get_database_pool(engine_name)
    return await pool.execute_single(query, *args)

async def db_command(engine_name: str, command: str, *args) -> str:
    """Execute a database command for an engine"""
    pool = await get_database_pool(engine_name)
    return await pool.execute_command(command, *args)


# Test function
async def test_connection_pool():
    """Test the database connection pool"""
    try:
        # Test analytics engine pool
        analytics_pool = await get_database_pool("analytics")
        result = await analytics_pool.execute_single(
            "SELECT 'Analytics Engine Connected' as message, current_timestamp as time"
        )
        print(f"✅ Analytics Pool Test: {result}")
        
        # Test risk engine pool  
        risk_pool = await get_database_pool("risk")
        result = await risk_pool.execute_single(
            "SELECT 'Risk Engine Connected' as message, current_timestamp as time"
        )
        print(f"✅ Risk Pool Test: {result}")
        
        # Get pool statistics
        analytics_stats = await analytics_pool.get_pool_stats()
        risk_stats = await risk_pool.get_pool_stats()
        
        print(f"📊 Analytics Pool: {analytics_stats}")
        print(f"📊 Risk Pool: {risk_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection_pool())
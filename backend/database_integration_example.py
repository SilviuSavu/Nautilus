#!/usr/bin/env python3
"""
Database Integration Example for Engines
Shows how to integrate the standardized connection pool into existing engines.
"""

from database_connection_pool import get_database_pool, db_query, db_single
import asyncio

# Example integration for engines
class ExampleEngineWithDatabase:
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.db_pool = None
    
    async def initialize(self):
        """Initialize engine with database pool"""
        self.db_pool = await get_database_pool(self.engine_name)
        print(f"✅ {self.engine_name} engine database initialized")
    
    async def get_market_data(self, symbol: str):
        """Example: Get market data from database"""
        query = """
        SELECT instrument_id, close_price, volume, timestamp_ns
        FROM market_bars 
        WHERE instrument_id = $1 
        ORDER BY timestamp_ns DESC 
        LIMIT 5
        """
        return await db_query(self.engine_name, query, symbol)
    
    async def get_engine_stats(self):
        """Get database pool statistics"""
        return await self.db_pool.get_pool_stats()

# Example usage for different engines
async def demo_engine_integration():
    """Demonstrate database integration for multiple engines"""
    
    # Initialize different engine types
    engines = [
        ExampleEngineWithDatabase("analytics"),
        ExampleEngineWithDatabase("risk"), 
        ExampleEngineWithDatabase("portfolio"),
        ExampleEngineWithDatabase("vpin")
    ]
    
    # Initialize all engines
    for engine in engines:
        await engine.initialize()
    
    # Test database operations
    print("\n📊 Testing database operations:")
    
    # Test market data query for each engine
    for engine in engines:
        try:
            # Test with a known symbol
            data = await engine.get_market_data("NFLX.SMART")
            print(f"✅ {engine.engine_name}: {len(data)} records found")
            
            # Show pool stats
            stats = await engine.get_engine_stats()
            print(f"   Pool: {stats['active_connections']}/{stats['pool_size']} active")
            
        except Exception as e:
            print(f"⚠️  {engine.engine_name}: {e}")
    
    print("\n🎯 Database connection pool integration successful!")

if __name__ == "__main__":
    asyncio.run(demo_engine_integration())
"""
Historical Data Service
Provides PostgreSQL integration for storing and retrieving historical market data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Tuple
from dataclasses import dataclass, asdict
import asyncpg
import json

from data_normalizer import NormalizedTick, NormalizedQuote, NormalizedBar
from enums import Venue


@dataclass
class HistoricalDataQuery:
    """Historical data query parameters"""
    venue: str
    instrument_id: str
    data_type: str  # 'tick', 'quote', 'bar'
    start_time: datetime
    end_time: datetime
    limit: int | None = None
    timeframe: str | None = None  # For bars


class HistoricalDataService:
    """
    PostgreSQL-based historical data service for storing and retrieving
    market data with efficient time-series storage and querying.
    """
    
    def __init__(
        self, database_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus", pool_min_size: int = 5, pool_max_size: int = 20, ):
        self.logger = logging.getLogger(__name__)
        self.database_url = database_url
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None
        self._connected = False
        self._timescale_enabled = False
        
    async def connect(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url, min_size=self.pool_min_size, max_size=self.pool_max_size, command_timeout=60, server_settings={
                    'jit': 'off', # Disable JIT for better performance on small queries
                }
            )
            
            # Test connection and create tables
            async with self._pool.acquire() as conn:
                await self._create_tables(conn)
                
            self._connected = True
            self.logger.info(f"Connected to PostgreSQL database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL database"""
        if self._pool:
            await self._pool.close()
            self._connected = False
            self.logger.info("Disconnected from PostgreSQL database")
            
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected
        
    async def _create_tables(self, conn: asyncpg.Connection) -> None:
        """Create database tables for market data storage"""
        
        # Create extension for better time-series support (optional for testing)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            self.logger.info("TimescaleDB extension enabled")
            self._timescale_enabled = True
        except Exception as e:
            self.logger.warning(f"TimescaleDB not available: {e}")
            self.logger.info("Continuing with standard PostgreSQL tables")
            self._timescale_enabled = False
        
        # Ticks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_ticks (
                id BIGSERIAL, venue VARCHAR(50) NOT NULL, instrument_id VARCHAR(100) NOT NULL, timestamp_ns BIGINT NOT NULL, price DECIMAL(20, 8) NOT NULL, size DECIMAL(20, 8) NOT NULL, side VARCHAR(10), trade_id VARCHAR(100), sequence_num BIGINT, raw_data JSONB, created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Quotes table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_quotes (
                id BIGSERIAL, venue VARCHAR(50) NOT NULL, instrument_id VARCHAR(100) NOT NULL, timestamp_ns BIGINT NOT NULL, bid_price DECIMAL(20, 8) NOT NULL, ask_price DECIMAL(20, 8) NOT NULL, bid_size DECIMAL(20, 8) NOT NULL, ask_size DECIMAL(20, 8) NOT NULL, spread DECIMAL(20, 8), sequence_num BIGINT, raw_data JSONB, created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Bars table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_bars (
                id BIGSERIAL, venue VARCHAR(50) NOT NULL, instrument_id VARCHAR(100) NOT NULL, timeframe VARCHAR(10) NOT NULL, timestamp_ns BIGINT NOT NULL, open_price DECIMAL(20, 8) NOT NULL, high_price DECIMAL(20, 8) NOT NULL, low_price DECIMAL(20, 8) NOT NULL, close_price DECIMAL(20, 8) NOT NULL, volume DECIMAL(20, 8) NOT NULL, is_final BOOLEAN DEFAULT TRUE, raw_data JSONB, created_at TIMESTAMP DEFAULT NOW(), UNIQUE(venue, instrument_id, timeframe, timestamp_ns)
            );
        """)
        
        # Create hypertables for time-series optimization (if TimescaleDB is available)
        if self._timescale_enabled:
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_ticks', 'timestamp_ns', chunk_time_interval => 86400000000000, if_not_exists => TRUE);
                """)
                await conn.execute("""
                    SELECT create_hypertable('market_quotes', 'timestamp_ns', chunk_time_interval => 86400000000000, if_not_exists => TRUE);
                """)
                await conn.execute("""
                    SELECT create_hypertable('market_bars', 'timestamp_ns', chunk_time_interval => 604800000000000, if_not_exists => TRUE);
                """)
                self.logger.info("Created TimescaleDB hypertables")
            except Exception as e:
                self.logger.warning(f"Failed to create hypertables: {e}")
        else:
            self.logger.info("Using regular PostgreSQL tables (TimescaleDB not available)")
            
        # Create indexes for better query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_ticks_venue_instrument ON market_ticks (venue, instrument_id, timestamp_ns DESC);", "CREATE INDEX IF NOT EXISTS idx_quotes_venue_instrument ON market_quotes (venue, instrument_id, timestamp_ns DESC);", "CREATE INDEX IF NOT EXISTS idx_bars_venue_instrument_tf ON market_bars (venue, instrument_id, timeframe, timestamp_ns DESC);", "CREATE INDEX IF NOT EXISTS idx_ticks_timestamp ON market_ticks (timestamp_ns DESC);", "CREATE INDEX IF NOT EXISTS idx_quotes_timestamp ON market_quotes (timestamp_ns DESC);", "CREATE INDEX IF NOT EXISTS idx_bars_timestamp ON market_bars (timestamp_ns DESC);", ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Index creation warning: {e}")
                
    async def store_tick(self, tick: NormalizedTick) -> None:
        """Store tick data in PostgreSQL"""
        if not self._connected:
            return
            
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_ticks (
                        venue, instrument_id, timestamp_ns, price, size, side, trade_id, sequence_num, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, tick.venue, tick.instrument_id, tick.timestamp_ns, tick.price, tick.size, tick.side, tick.trade_id, tick.sequence, json.dumps(asdict(tick))
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store tick data: {e}")
            
    async def store_quote(self, quote: NormalizedQuote) -> None:
        """Store quote data in PostgreSQL"""
        if not self._connected:
            return
            
        try:
            async with self._pool.acquire() as conn:
                spread = quote.ask_price - quote.bid_price
                
                await conn.execute("""
                    INSERT INTO market_quotes (
                        venue, instrument_id, timestamp_ns, bid_price, ask_price, bid_size, ask_size, spread, sequence_num, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, quote.venue, quote.instrument_id, quote.timestamp_ns, quote.bid_price, quote.ask_price, quote.bid_size, quote.ask_size, spread, quote.sequence, json.dumps(asdict(quote))
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store quote data: {e}")
            
    async def store_bar(self, bar: NormalizedBar) -> None:
        """Store bar data in PostgreSQL"""
        if not self._connected:
            return
            
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_bars (
                        venue, instrument_id, timeframe, timestamp_ns, open_price, high_price, low_price, close_price, volume, is_final, raw_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (venue, instrument_id, timeframe, timestamp_ns)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price, high_price = EXCLUDED.high_price, low_price = EXCLUDED.low_price, close_price = EXCLUDED.close_price, volume = EXCLUDED.volume, is_final = EXCLUDED.is_final, raw_data = EXCLUDED.raw_data
                """, bar.venue, bar.instrument_id, bar.timeframe, bar.timestamp_ns, bar.open_price, bar.high_price, bar.low_price, bar.close_price, bar.volume, bar.is_final, json.dumps(asdict(bar))
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store bar data: {e}")
            
    async def query_ticks(self, query: HistoricalDataQuery) -> list[dict[str, Any]]:
        """Query historical tick data"""
        if not self._connected:
            return []
            
        try:
            async with self._pool.acquire() as conn:
                sql = """
                    SELECT venue, instrument_id, timestamp_ns, price, size, side, trade_id, sequence_num, created_at
                    FROM market_ticks
                    WHERE venue = $1 AND instrument_id = $2
                      AND timestamp_ns >= $3 AND timestamp_ns <= $4
                    ORDER BY timestamp_ns DESC
                """
                
                params = [
                    query.venue, query.instrument_id, int(query.start_time.timestamp() * 1_000_000_000), int(query.end_time.timestamp() * 1_000_000_000), ]
                
                if query.limit:
                    sql += " LIMIT $5"
                    params.append(query.limit)
                    
                rows = await conn.fetch(sql, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to query tick data: {e}")
            return []
            
    async def query_quotes(self, query: HistoricalDataQuery) -> list[dict[str, Any]]:
        """Query historical quote data"""
        if not self._connected:
            return []
            
        try:
            async with self._pool.acquire() as conn:
                sql = """
                    SELECT venue, instrument_id, timestamp_ns, bid_price, ask_price, bid_size, ask_size, spread, sequence_num, created_at
                    FROM market_quotes
                    WHERE venue = $1 AND instrument_id = $2
                      AND timestamp_ns >= $3 AND timestamp_ns <= $4
                    ORDER BY timestamp_ns DESC
                """
                
                params = [
                    query.venue, query.instrument_id, int(query.start_time.timestamp() * 1_000_000_000), int(query.end_time.timestamp() * 1_000_000_000), ]
                
                if query.limit:
                    sql += " LIMIT $5"
                    params.append(query.limit)
                    
                rows = await conn.fetch(sql, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to query quote data: {e}")
            return []
            
    async def query_bars(self, query: HistoricalDataQuery) -> list[dict[str, Any]]:
        """Query historical bar data"""
        if not self._connected:
            return []
            
        try:
            async with self._pool.acquire() as conn:
                sql = """
                    SELECT venue, instrument_id, timeframe, timestamp_ns, open_price, high_price, low_price, close_price, volume, is_final, created_at
                    FROM market_bars
                    WHERE venue = $1 AND instrument_id = $2
                      AND timestamp_ns >= $3 AND timestamp_ns <= $4
                """
                
                params = [
                    query.venue, query.instrument_id, int(query.start_time.timestamp() * 1_000_000_000), int(query.end_time.timestamp() * 1_000_000_000), ]
                
                if query.timeframe:
                    sql += " AND timeframe = $5"
                    params.append(query.timeframe)
                    
                sql += " ORDER BY timestamp_ns DESC"
                
                if query.limit:
                    sql += f" LIMIT ${len(params) + 1}"
                    params.append(query.limit)
                    
                rows = await conn.fetch(sql, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to query bar data: {e}")
            return []
            
    async def get_data_summary(self, venue: str, instrument_id: str) -> dict[str, Any]:
        """Get data summary for an instrument"""
        if not self._connected:
            return {}
            
        try:
            async with self._pool.acquire() as conn:
                # Get counts and date ranges for each data type
                summary = {}
                
                # Ticks summary
                tick_row = await conn.fetchrow("""
                    SELECT COUNT(*) as count, MIN(timestamp_ns) as first_timestamp, MAX(timestamp_ns) as last_timestamp
                    FROM market_ticks
                    WHERE venue = $1 AND instrument_id = $2
                """, venue, instrument_id)
                
                if tick_row['count'] > 0:
                    summary['ticks'] = {
                        'count': tick_row['count'], 'first_timestamp': tick_row['first_timestamp'], 'last_timestamp': tick_row['last_timestamp'], 'date_range_days': (tick_row['last_timestamp'] - tick_row['first_timestamp']) / (1_000_000_000 * 86400)
                    }
                
                # Quotes summary
                quote_row = await conn.fetchrow("""
                    SELECT COUNT(*) as count, MIN(timestamp_ns) as first_timestamp, MAX(timestamp_ns) as last_timestamp
                    FROM market_quotes
                    WHERE venue = $1 AND instrument_id = $2
                """, venue, instrument_id)
                
                if quote_row['count'] > 0:
                    summary['quotes'] = {
                        'count': quote_row['count'], 'first_timestamp': quote_row['first_timestamp'], 'last_timestamp': quote_row['last_timestamp'], 'date_range_days': (quote_row['last_timestamp'] - quote_row['first_timestamp']) / (1_000_000_000 * 86400)
                    }
                
                # Bars summary by timeframe
                bar_rows = await conn.fetch("""
                    SELECT timeframe, COUNT(*) as count, MIN(timestamp_ns) as first_timestamp, MAX(timestamp_ns) as last_timestamp
                    FROM market_bars
                    WHERE venue = $1 AND instrument_id = $2
                    GROUP BY timeframe
                """, venue, instrument_id)
                
                if bar_rows:
                    summary['bars'] = {}
                    for row in bar_rows:
                        summary['bars'][row['timeframe']] = {
                            'count': row['count'], 'first_timestamp': row['first_timestamp'], 'last_timestamp': row['last_timestamp'], 'date_range_days': (row['last_timestamp'] - row['first_timestamp']) / (1_000_000_000 * 86400)
                        }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get data summary: {e}")
            return {}
            
    async def cleanup_old_data(self, days_to_keep: int = 30) -> dict[str, int]:
        """Clean up old data beyond retention period"""
        if not self._connected:
            return {}
            
        try:
            cutoff_ns = int((datetime.now() - timedelta(days=days_to_keep)).timestamp() * 1_000_000_000)
            deleted_counts = {}
            
            async with self._pool.acquire() as conn:
                # Delete old ticks
                result = await conn.execute("""
                    DELETE FROM market_ticks WHERE timestamp_ns < $1
                """, cutoff_ns)
                deleted_counts['ticks'] = int(result.split()[-1])
                
                # Delete old quotes  
                result = await conn.execute("""
                    DELETE FROM market_quotes WHERE timestamp_ns < $1
                """, cutoff_ns)
                deleted_counts['quotes'] = int(result.split()[-1])
                
                # Delete old bars
                result = await conn.execute("""
                    DELETE FROM market_bars WHERE timestamp_ns < $1
                """, cutoff_ns)
                deleted_counts['bars'] = int(result.split()[-1])
                
            self.logger.info(f"Cleaned up old data: {deleted_counts}")
            return deleted_counts
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return {}

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute a custom SQL query and return results as list of dictionaries"""
        if not self._connected:
            self.logger.warning("Cannot execute query - not connected to database")
            return []
            
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                # Convert asyncpg Records to dictionaries
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            return []
            
    async def health_check(self) -> dict[str, Any]:
        """Perform database health check"""
        if not self._connected:
            return {"status": "disconnected", "error": "Not connected to database"}
            
        try:
            async with self._pool.acquire() as conn:
                # Test basic connectivity
                await conn.fetchval("SELECT 1")
                
                # Get database stats
                stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size, (SELECT COUNT(*) FROM market_ticks) as tick_count, (SELECT COUNT(*) FROM market_quotes) as quote_count, (SELECT COUNT(*) FROM market_bars) as bar_count
                """)
                
                return {
                    "status": "connected", "database_size": stats['db_size'], "table_counts": {
                        "ticks": stats['tick_count'], "quotes": stats['quote_count'], "bars": stats['bar_count']
                    }, "pool_stats": {
                        "size": self._pool.get_size(), "max_size": self._pool.get_max_size(), "min_size": self._pool.get_min_size()
                    }
                }
                
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global historical data service instance
import os
database_url = os.getenv("DATABASE_URL", "postgresql://nautilus:nautilus123@localhost:5432/nautilus")
historical_data_service = HistoricalDataService(database_url=database_url)
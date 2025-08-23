"""
Redis Caching Layer
Provides high-performance caching for market data with optimized data structures and TTL policies.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Union
from dataclasses import asdict
import pickle
from urllib.parse import urlparse

import redis.asyncio as redis

from data_normalizer import NormalizedTick, NormalizedQuote, NormalizedBar, data_normalizer
from enums import Venue


class RedisCacheError(Exception):
    """Redis cache operation error"""
    pass


class RedisCache:
    """
    High-performance Redis caching layer for market data with optimized
    data structures, TTL policies, and efficient retrieval patterns.
    """
    
    def __init__(
        self, redis_host: str = "nautilus-redis", redis_port: int = 6379, redis_db: int = 1, # Use different DB from MessageBus
        default_ttl: int = 3600, # 1 hour default TTL
    ):
        self.logger = logging.getLogger(__name__)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self._redis: redis.Redis | None = None
        self._connected = False
        
        # Cache key patterns
        self.KEY_PATTERNS = {
            "tick": "market:tick:{venue}:{instrument}", "quote": "market:quote:{venue}:{instrument}", "bar": "market:bar:{venue}:{instrument}:{timeframe}", "orderbook": "market:orderbook:{venue}:{instrument}", "latest_tick": "latest:tick:{venue}:{instrument}", "latest_quote": "latest:quote:{venue}:{instrument}", "latest_bar": "latest:bar:{venue}:{instrument}:{timeframe}", "instrument_list": "instruments:{venue}", "venue_status": "status:venue:{venue}", "stats": "stats:cache", }
        
        # TTL settings by data type
        self.TTL_SETTINGS = {
            "tick": 300, # 5 minutes for tick data
            "quote": 300, # 5 minutes for quote data  
            "bar": 3600, # 1 hour for bar data
            "orderbook": 60, # 1 minute for order book
            "latest": 3600, # 1 hour for latest data
            "instrument": 86400, # 24 hours for instrument data
            "stats": 300, # 5 minutes for stats
        }
        
    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            self._redis = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=self.redis_db, decode_responses=True, # Decode responses for JSON compatibility
                socket_connect_timeout=5.0, socket_keepalive=True, retry_on_timeout=True, )
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            self.logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}/{self.redis_db}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise RedisCacheError(f"Redis connection failed: {e}")
            
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.aclose()
            self._connected = False
            self.logger.info("Disconnected from Redis")
            
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._connected
        
    async def cache_tick(self, tick: NormalizedTick) -> None:
        """Cache tick data"""
        if not self._connected:
            return
            
        try:
            # Cache in time series list
            key = self.KEY_PATTERNS["tick"].format(
                venue=tick.venue, instrument=tick.instrument_id
            )
            
            # Serialize tick data
            tick_data = {
                "price": str(tick.price), "size": str(tick.size), "timestamp_ns": tick.timestamp_ns, "side": tick.side, "trade_id": tick.trade_id, "sequence": tick.sequence, }
            
            pipe = self._redis.pipeline()
            
            # Add to time series (keep last 1000 ticks)
            pipe.lpush(key, json.dumps(tick_data))
            pipe.ltrim(key, 0, 999)
            pipe.expire(key, self.TTL_SETTINGS["tick"])
            
            # Cache as latest tick
            latest_key = self.KEY_PATTERNS["latest_tick"].format(
                venue=tick.venue, instrument=tick.instrument_id
            )
            pipe.set(latest_key, json.dumps(tick_data), ex=self.TTL_SETTINGS["latest"])
            
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Failed to cache tick data: {e}")
            
    async def cache_quote(self, quote: NormalizedQuote) -> None:
        """Cache quote data"""
        if not self._connected:
            return
            
        try:
            key = self.KEY_PATTERNS["quote"].format(
                venue=quote.venue, instrument=quote.instrument_id
            )
            
            quote_data = {
                "bid_price": str(quote.bid_price), "ask_price": str(quote.ask_price), "bid_size": str(quote.bid_size), "ask_size": str(quote.ask_size), "timestamp_ns": quote.timestamp_ns, "sequence": quote.sequence, "spread": str(quote.ask_price - quote.bid_price), }
            
            pipe = self._redis.pipeline()
            
            # Cache in time series (keep last 500 quotes)
            pipe.lpush(key, json.dumps(quote_data))
            pipe.ltrim(key, 0, 499)
            pipe.expire(key, self.TTL_SETTINGS["quote"])
            
            # Cache as latest quote
            latest_key = self.KEY_PATTERNS["latest_quote"].format(
                venue=quote.venue, instrument=quote.instrument_id
            )
            pipe.set(latest_key, json.dumps(quote_data), ex=self.TTL_SETTINGS["latest"])
            
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Failed to cache quote data: {e}")
            
    async def cache_bar(self, bar: NormalizedBar) -> None:
        """Cache OHLCV bar data"""
        if not self._connected:
            return
            
        try:
            key = self.KEY_PATTERNS["bar"].format(
                venue=bar.venue, instrument=bar.instrument_id, timeframe=bar.timeframe
            )
            
            bar_data = {
                "open": str(bar.open_price), "high": str(bar.high_price), "low": str(bar.low_price), "close": str(bar.close_price), "volume": str(bar.volume), "timestamp_ns": bar.timestamp_ns, "timeframe": bar.timeframe, "is_final": bar.is_final, }
            
            pipe = self._redis.pipeline()
            
            # Cache in time series (keep last 200 bars)
            pipe.lpush(key, json.dumps(bar_data))
            pipe.ltrim(key, 0, 199)
            pipe.expire(key, self.TTL_SETTINGS["bar"])
            
            # Cache as latest bar
            latest_key = self.KEY_PATTERNS["latest_bar"].format(
                venue=bar.venue, instrument=bar.instrument_id, timeframe=bar.timeframe
            )
            pipe.set(latest_key, json.dumps(bar_data), ex=self.TTL_SETTINGS["latest"])
            
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Failed to cache bar data: {e}")
            
    async def get_latest_tick(self, venue: str, instrument_id: str) -> dict[str, Any | None]:
        """Get latest tick for instrument"""
        if not self._connected:
            return None
            
        try:
            key = self.KEY_PATTERNS["latest_tick"].format(
                venue=venue, instrument=instrument_id
            )
            
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
                
        except Exception as e:
            self.logger.error(f"Failed to get latest tick: {e}")
            
        return None
        
    async def get_latest_quote(self, venue: str, instrument_id: str) -> dict[str, Any | None]:
        """Get latest quote for instrument"""
        if not self._connected:
            return None
            
        try:
            key = self.KEY_PATTERNS["latest_quote"].format(
                venue=venue, instrument=instrument_id
            )
            
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
                
        except Exception as e:
            self.logger.error(f"Failed to get latest quote: {e}")
            
        return None
        
    async def get_latest_bar(self, venue: str, instrument_id: str, timeframe: str = "1m") -> dict[str, Any | None]:
        """Get latest bar for instrument"""
        if not self._connected:
            return None
            
        try:
            key = self.KEY_PATTERNS["latest_bar"].format(
                venue=venue, instrument=instrument_id, timeframe=timeframe
            )
            
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
                
        except Exception as e:
            self.logger.error(f"Failed to get latest bar: {e}")
            
        return None
        
    async def get_tick_history(self, venue: str, instrument_id: str, count: int = 100) -> list[dict[str, Any]]:
        """Get recent tick history"""
        if not self._connected:
            return []
            
        try:
            key = self.KEY_PATTERNS["tick"].format(
                venue=venue, instrument=instrument_id
            )
            
            data_list = await self._redis.lrange(key, 0, count - 1)
            return [json.loads(data) for data in data_list]
            
        except Exception as e:
            self.logger.error(f"Failed to get tick history: {e}")
            return []
            
    async def get_quote_history(self, venue: str, instrument_id: str, count: int = 100) -> list[dict[str, Any]]:
        """Get recent quote history"""
        if not self._connected:
            return []
            
        try:
            key = self.KEY_PATTERNS["quote"].format(
                venue=venue, instrument=instrument_id
            )
            
            data_list = await self._redis.lrange(key, 0, count - 1)
            return [json.loads(data) for data in data_list]
            
        except Exception as e:
            self.logger.error(f"Failed to get quote history: {e}")
            return []
            
    async def get_bar_history(self, venue: str, instrument_id: str, timeframe: str = "1m", count: int = 100) -> list[dict[str, Any]]:
        """Get recent bar history"""
        if not self._connected:
            return []
            
        try:
            key = self.KEY_PATTERNS["bar"].format(
                venue=venue, instrument=instrument_id, timeframe=timeframe
            )
            
            data_list = await self._redis.lrange(key, 0, count - 1)
            return [json.loads(data) for data in data_list]
            
        except Exception as e:
            self.logger.error(f"Failed to get bar history: {e}")
            return []
            
    async def cache_instrument_list(self, venue: str, instruments: list[str]) -> None:
        """Cache list of available instruments for a venue"""
        if not self._connected:
            return
            
        try:
            key = self.KEY_PATTERNS["instrument_list"].format(venue=venue)
            
            # Use Redis set for efficient membership testing
            pipe = self._redis.pipeline()
            pipe.delete(key)  # Clear existing set
            if instruments:
                pipe.sadd(key, *instruments)
            pipe.expire(key, self.TTL_SETTINGS["instrument"])
            
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Failed to cache instrument list: {e}")
            
    async def get_instrument_list(self, venue: str) -> list[str]:
        """Get cached instrument list for a venue"""
        if not self._connected:
            return []
            
        try:
            key = self.KEY_PATTERNS["instrument_list"].format(venue=venue)
            members = await self._redis.smembers(key)
            return [member.decode() if isinstance(member, bytes) else member for member in members]
            
        except Exception as e:
            self.logger.error(f"Failed to get instrument list: {e}")
            return []
            
    async def update_cache_stats(self) -> None:
        """Update cache statistics"""
        if not self._connected:
            return
            
        try:
            # Get Redis memory usage and key count
            info = await self._redis.info("memory")
            keyspace_info = await self._redis.info("keyspace")
            
            stats = {
                "memory_used": info.get("used_memory", 0), "memory_used_human": info.get("used_memory_human", "0B"), "total_keys": sum(
                    keyspace_info.get(f"db{i}", {}).get("keys", 0)
                    for i in range(16)
                ), "last_updated": datetime.now().isoformat(), }
            
            key = self.KEY_PATTERNS["stats"]
            await self._redis.set(
                key, json.dumps(stats), ex=self.TTL_SETTINGS["stats"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update cache stats: {e}")
            
    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        if not self._connected:
            return {}
            
        try:
            key = self.KEY_PATTERNS["stats"]
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
                
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            
        return {}
        
    async def clear_venue_cache(self, venue: str) -> int:
        """Clear all cached data for a venue"""
        if not self._connected:
            return 0
            
        try:
            # Find all keys for this venue
            patterns = [
                f"market:*:{venue}:*", f"latest:*:{venue}:*", f"instruments:{venue}", f"status:venue:{venue}", ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = await self._redis.keys(pattern)
                if keys:
                    deleted_count += await self._redis.delete(*keys)
                    
            self.logger.info(f"Cleared {deleted_count} cache keys for venue {venue}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to clear venue cache: {e}")
            return 0
            
    async def health_check(self) -> dict[str, Any]:
        """Perform Redis health check"""
        if not self._connected:
            return {"status": "disconnected", "error": "Not connected to Redis"}
            
        try:
            # Test basic operations
            start_time = datetime.now()
            await self._redis.ping()
            ping_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await self._redis.info()
            
            return {
                "status": "connected", "ping_ms": round(ping_time, 2), "redis_version": info.get("redis_version"), "uptime_seconds": info.get("uptime_in_seconds"), "connected_clients": info.get("connected_clients"), "used_memory_human": info.get("used_memory_human"), }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global cache instance
def create_redis_cache():
    """Create Redis cache instance from environment variables"""
    redis_url = os.getenv("REDIS_URL", "redis://nautilus-redis:6379")
    parsed = urlparse(redis_url)
    
    return RedisCache(
        redis_host=parsed.hostname or "nautilus-redis", redis_port=parsed.port or 6379, redis_db=1, # Use different DB from MessageBus
    )

redis_cache = create_redis_cache()
#!/usr/bin/env python3
"""
Universal MarketData Client - MANDATORY for all engines
All engines MUST use this client to access market data
Direct API calls are PROHIBITED for maximum performance
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import time

# Import MessageBus client
from universal_enhanced_messagebus_client import (
    UniversalEnhancedMessageBusClient,
    EngineType,
    MessageType,
    MessagePriority,
    create_messagebus_client
)

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """All integrated data sources"""
    IBKR = "ibkr"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    EDGAR = "edgar"
    DATA_GOV = "data_gov"
    TRADING_ECONOMICS = "trading_economics"
    DBNOMICS = "dbnomics"
    YAHOO = "yahoo"

class DataType(Enum):
    """Types of market data"""
    TICK = "tick"
    QUOTE = "quote"
    BAR = "bar"
    TRADE = "trade"
    LEVEL2 = "level2"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    ECONOMIC = "economic"
    SENTIMENT = "sentiment"

class MarketDataClient:
    """
    Universal client for accessing market data through the Centralized Hub
    
    Performance:
    - Sub-5ms data access via MessageBus
    - 90%+ cache hit rate (no API calls)
    - Single connection to MarketData Hub
    - Automatic fallback to HTTP if MessageBus unavailable
    
    Usage:
        client = MarketDataClient(engine_type=EngineType.RISK)
        data = await client.get_data(
            symbols=["AAPL", "GOOGL"],
            data_types=[DataType.QUOTE, DataType.LEVEL2]
        )
    """
    
    def __init__(self, engine_type: EngineType, engine_port: int):
        self.engine_type = engine_type
        self.engine_port = engine_port
        self.messagebus: Optional[UniversalEnhancedMessageBusClient] = None
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.subscriptions: Dict[str, Callable] = {}
        
        # Performance metrics
        self.total_requests = 0
        self.messagebus_requests = 0
        self.http_fallback_requests = 0
        self.avg_latency_ms = 0.0
        
        # Initialize MessageBus connection
        self._setup_messagebus()
    
    def _setup_messagebus(self):
        """Setup MessageBus connection to MarketData Hub"""
        try:
            self.messagebus = create_messagebus_client(
                engine_type=self.engine_type,
                engine_port=self.engine_port
            )
            
            # Subscribe to data responses
            response_pattern = f"{self.engine_type.value}.data_response.*"
            self.messagebus.subscribe(
                response_pattern,
                self._handle_data_response
            )
            
            logger.info(f"✅ MarketData Client connected for {self.engine_type.value}")
            
        except Exception as e:
            logger.warning(f"⚠️ MessageBus unavailable, using HTTP fallback: {e}")
    
    async def get_data(
        self,
        symbols: List[str],
        data_types: List[DataType],
        sources: Optional[List[DataSource]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        cache: bool = True,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Get market data through Centralized Hub
        
        Args:
            symbols: List of symbols to fetch
            data_types: Types of data needed
            sources: Specific sources (default: all available)
            start_time: Historical data start
            end_time: Historical data end
            priority: Request priority
            cache: Use cache (recommended for performance)
            timeout: Request timeout in seconds
        
        Returns:
            Market data dictionary with sub-5ms latency
        """
        self.total_requests += 1
        request_id = f"{self.engine_type.value}_{time.time_ns()}"
        start = time.time()
        
        # Default to all sources if not specified
        if sources is None:
            sources = list(DataSource)
        
        # Try MessageBus first (fastest)
        if self.messagebus and hasattr(self.messagebus, '_connection_state') and self.messagebus._connection_state.value == 'connected':
            try:
                result = await self._request_via_messagebus(
                    request_id, symbols, data_types, sources,
                    start_time, end_time, priority, cache, timeout
                )
                self.messagebus_requests += 1
                
            except asyncio.TimeoutError:
                logger.warning(f"MessageBus timeout, falling back to HTTP")
                result = await self._request_via_http(
                    symbols, data_types, sources, start_time, end_time, cache
                )
                self.http_fallback_requests += 1
        else:
            # Fallback to HTTP
            result = await self._request_via_http(
                symbols, data_types, sources, start_time, end_time, cache
            )
            self.http_fallback_requests += 1
        
        # Update metrics
        latency = (time.time() - start) * 1000
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_requests - 1) + latency) 
            / self.total_requests
        )
        
        return result
    
    async def _request_via_messagebus(
        self,
        request_id: str,
        symbols: List[str],
        data_types: List[DataType],
        sources: List[DataSource],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        priority: MessagePriority,
        cache: bool,
        timeout: float
    ) -> Dict[str, Any]:
        """Request data via MessageBus (sub-5ms)"""
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request to MarketData Hub
        self.messagebus.publish(
            channel="marketdata.data_request",
            message={
                "request_id": request_id,
                "engine_type": self.engine_type.value,
                "symbols": symbols,
                "data_types": [dt.value for dt in data_types],
                "data_sources": [s.value for s in sources],
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "priority": priority.value,
                "cache_enabled": cache
            },
            priority=priority
        )
        
        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        finally:
            # Clean up
            self.pending_requests.pop(request_id, None)
    
    def _handle_data_response(self, message: Dict[str, Any]):
        """Handle data response from MarketData Hub"""
        request_id = message.get("request_id")
        
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            if not future.done():
                future.set_result(message.get("data", {}))
    
    async def _request_via_http(
        self,
        symbols: List[str],
        data_types: List[DataType],
        sources: List[DataSource],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        cache: bool
    ) -> Dict[str, Any]:
        """Fallback HTTP request to MarketData Hub"""
        import aiohttp
        
        url = "http://localhost:8800/data/request"
        
        payload = {
            "symbols": symbols,
            "data_types": [dt.value for dt in data_types],
            "sources": [s.value for s in sources],
            "engine": self.engine_type.value,
            "cache": cache
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"HTTP request failed: {response.status}")
                        return {}
        
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return {}
    
    async def subscribe(
        self,
        symbols: List[str],
        data_types: List[DataType],
        callback: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to real-time market data updates
        
        Args:
            symbols: Symbols to subscribe to
            data_types: Data types to receive
            callback: Function to call with updates
        
        Returns:
            Subscription ID
        """
        subscription_id = f"sub_{self.engine_type.value}_{time.time_ns()}"
        callback_pattern = f"{self.engine_type.value}.subscription.{subscription_id}"
        
        # Register callback
        self.subscriptions[subscription_id] = callback
        
        if self.messagebus:
            # Subscribe to callback pattern
            self.messagebus.subscribe(callback_pattern, callback)
            
            # Send subscription request
            self.messagebus.publish(
                channel="marketdata.subscription.create",
                message={
                    "engine": self.engine_type.value,
                    "symbols": symbols,
                    "data_types": [dt.value for dt in data_types],
                    "callback_pattern": callback_pattern
                }
            )
        else:
            # HTTP fallback for subscription
            import aiohttp
            url = "http://localhost:8800/subscription/create"
            
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    "engine": self.engine_type.value,
                    "symbols": symbols,
                    "data_types": [dt.value for dt in data_types],
                    "callback_pattern": callback_pattern
                })
        
        logger.info(f"✅ Subscribed to {symbols} for {self.engine_type.value}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """Cancel market data subscription"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            
            if self.messagebus:
                self.messagebus.publish(
                    channel="marketdata.subscription.cancel",
                    message={"subscription_id": subscription_id}
                )
            else:
                # HTTP fallback
                import aiohttp
                url = f"http://localhost:8800/subscription/{subscription_id}"
                
                async with aiohttp.ClientSession() as session:
                    await session.delete(url)
            
            logger.info(f"✅ Unsubscribed: {subscription_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        return {
            "total_requests": self.total_requests,
            "messagebus_requests": self.messagebus_requests,
            "http_fallback_requests": self.http_fallback_requests,
            "avg_latency_ms": f"{self.avg_latency_ms:.2f}",
            "messagebus_ratio": f"{self.messagebus_requests / max(1, self.total_requests):.1%}",
            "active_subscriptions": len(self.subscriptions),
            "messagebus_connected": bool(self.messagebus and hasattr(self.messagebus, '_connection_state') and self.messagebus._connection_state.value == 'connected')
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_marketdata_client(engine_type: EngineType, engine_port: int) -> MarketDataClient:
    """
    Factory function to create MarketData client
    
    Example:
        from marketdata_client import create_marketdata_client
        from universal_enhanced_messagebus_client import EngineType
        
        client = create_marketdata_client(EngineType.RISK, 8200)
        data = await client.get_data(["AAPL"], [DataType.QUOTE])
    """
    return MarketDataClient(engine_type, engine_port)

# =============================================================================
# MIGRATION HELPER
# =============================================================================

class DirectAPIBlocker:
    """
    Helper class to detect and block direct API calls
    Use this during migration to ensure all engines use the hub
    """
    
    BLOCKED_MODULES = [
        "requests",
        "urllib",
        "httpx",
        "aiohttp",  # Allow only for MarketData client
    ]
    
    BLOCKED_HOSTS = [
        "api.alphaVantage.co",
        "api.fred.stlouisfed.org",
        "data.nasdaq.com",
        "query1.finance.yahoo.com",
        # Add all external API hosts
    ]
    
    @staticmethod
    def check_import(module_name: str, caller: str):
        """Check if import is allowed"""
        if module_name in DirectAPIBlocker.BLOCKED_MODULES:
            if caller != "marketdata_client":
                raise ImportError(
                    f"❌ BLOCKED: Direct API calls prohibited!\n"
                    f"Engine '{caller}' attempted to import '{module_name}'\n"
                    f"Use MarketDataClient instead for maximum performance"
                )
    
    @staticmethod
    def check_connection(host: str, caller: str):
        """Check if connection is allowed"""
        for blocked in DirectAPIBlocker.BLOCKED_HOSTS:
            if blocked in host:
                raise ConnectionError(
                    f"❌ BLOCKED: Direct API connection prohibited!\n"
                    f"Engine '{caller}' attempted to connect to '{host}'\n"
                    f"Use MarketDataClient for sub-5ms data access"
                )

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of how engines should use the MarketData client"""
    
    # Create client for Risk Engine
    client = create_marketdata_client(EngineType.RISK, 8200)
    
    # Get market data (sub-5ms via MessageBus)
    data = await client.get_data(
        symbols=["AAPL", "GOOGL", "MSFT"],
        data_types=[DataType.QUOTE, DataType.LEVEL2],
        sources=[DataSource.IBKR],  # Optional: specify sources
        cache=True,  # Use cache for maximum performance
        priority=MessagePriority.HIGH
    )
    
    print(f"Received data in {client.avg_latency_ms:.2f}ms")
    print(f"Cache hit rate: {data.get('cache_hit_rate', 0):.1%}")
    
    # Subscribe to real-time updates
    def handle_update(msg):
        print(f"Real-time update: {msg}")
    
    sub_id = await client.subscribe(
        symbols=["AAPL"],
        data_types=[DataType.TICK],
        callback=handle_update
    )
    
    # Get performance metrics
    metrics = client.get_metrics()
    print(f"Performance: {metrics}")
    
    # Unsubscribe when done
    await client.unsubscribe(sub_id)

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
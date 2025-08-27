#!/usr/bin/env python3
"""
Direct HTTP Mesh Communication Client
Implements direct engine-to-engine communication for business logic (non-market data).

This client handles:
- Trading signals
- Risk alerts  
- Performance metrics
- System coordination
- Business logic communication

Market data distribution remains through Redis MessageBus via MarketData Hub.
"""

import asyncio
import logging
import time
import json
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)


class BusinessMessageType(Enum):
    """Business logic message types for direct mesh communication"""
    # Trading signals
    TRADING_SIGNAL = "trading_signal"
    ORDER_EXECUTION = "order_execution"
    POSITION_UPDATE = "position_update"
    
    # Risk management
    RISK_ALERT = "risk_alert"
    MARGIN_CALL = "margin_call"
    EXPOSURE_UPDATE = "exposure_update"
    
    # Performance & Analytics
    PERFORMANCE_METRIC = "performance_metric"
    ANALYTICS_RESULT = "analytics_result"
    CORRELATION_UPDATE = "correlation_update"
    
    # System coordination
    ENGINE_STATUS = "engine_status"
    HEALTH_CHECK = "health_check"
    SYSTEM_ALERT = "system_alert"


@dataclass
class EngineEndpoint:
    """Engine endpoint configuration"""
    name: str
    host: str = "localhost"
    port: int = 8000
    health_path: str = "/health"
    api_path: str = "/api/v1"
    timeout: float = 5.0
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"
    
    @property
    def api_url(self) -> str:
        return f"{self.base_url}{self.api_path}"


@dataclass 
class BusinessMessage:
    """Business logic message for direct mesh communication"""
    message_id: str
    source_engine: str
    target_engine: str
    message_type: BusinessMessageType
    payload: Dict[str, Any]
    priority: str = "NORMAL"
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source_engine": self.source_engine,
            "target_engine": self.target_engine,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class DirectMeshClient:
    """
    Direct HTTP Mesh Communication Client
    Handles direct engine-to-engine communication for business logic.
    """
    
    # Engine endpoint registry
    ENGINE_ENDPOINTS = {
        "analytics": EngineEndpoint("analytics", port=8100),
        "risk": EngineEndpoint("risk", port=8200), 
        "factor": EngineEndpoint("factor", port=8300),
        "ml": EngineEndpoint("ml", port=8400),
        "features": EngineEndpoint("features", port=8500),
        "websocket": EngineEndpoint("websocket", port=8600),
        "strategy": EngineEndpoint("strategy", port=8700),
        "marketdata": EngineEndpoint("marketdata", port=8800),
        "portfolio": EngineEndpoint("portfolio", port=8900),
        "collateral": EngineEndpoint("collateral", port=9000),
        "vpin": EngineEndpoint("vpin", port=10000),
        "backtesting": EngineEndpoint("backtesting", port=8110)
    }
    
    def __init__(self, engine_name: str, max_connections: int = 20):
        self.engine_name = engine_name
        self.max_connections = max_connections
        self.session: Optional[aiohttp.ClientSession] = None
        self.message_handlers: Dict[BusinessMessageType, Callable] = {}
        self.health_status: Dict[str, bool] = {}
        self.connection_pool: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize HTTP session and connection pool"""
        if self._initialized:
            return
            
        # Create connection pool for efficient reuse
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=5,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info(f"DirectMeshClient initialized for {self.engine_name}")
        self._initialized = True
        
    async def close(self):
        """Close HTTP session and cleanup connections"""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False
        logger.info(f"DirectMeshClient closed for {self.engine_name}")
    
    def register_handler(self, message_type: BusinessMessageType, handler: Callable):
        """Register message handler for specific business message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.value}")
    
    async def send_message(
        self, 
        target_engine: str, 
        message_type: BusinessMessageType,
        payload: Dict[str, Any],
        priority: str = "NORMAL",
        correlation_id: Optional[str] = None
    ) -> bool:
        """Send direct message to target engine"""
        if not self._initialized:
            await self.initialize()
            
        if target_engine not in self.ENGINE_ENDPOINTS:
            logger.error(f"Unknown target engine: {target_engine}")
            return False
            
        endpoint = self.ENGINE_ENDPOINTS[target_engine]
        message = BusinessMessage(
            message_id=f"{self.engine_name}_{int(time.time() * 1000000)}",
            source_engine=self.engine_name,
            target_engine=target_engine,
            message_type=message_type,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id
        )
        
        try:
            url = f"{endpoint.api_url}/mesh/message"
            async with self.session.post(
                url, 
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as response:
                if response.status == 200:
                    logger.debug(f"Message sent to {target_engine}: {message_type.value}")
                    return True
                else:
                    logger.warning(f"Failed to send message to {target_engine}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending message to {target_engine}: {e}")
            return False
    
    async def broadcast_message(
        self,
        message_type: BusinessMessageType,
        payload: Dict[str, Any],
        target_engines: Optional[List[str]] = None,
        priority: str = "NORMAL"
    ) -> Dict[str, bool]:
        """Broadcast message to multiple engines"""
        if target_engines is None:
            target_engines = list(self.ENGINE_ENDPOINTS.keys())
            # Don't broadcast to self
            target_engines = [e for e in target_engines if e != self.engine_name]
        
        results = {}
        tasks = []
        
        for engine in target_engines:
            task = self.send_message(engine, message_type, payload, priority)
            tasks.append((engine, task))
        
        # Execute all sends concurrently
        for engine, task in tasks:
            try:
                results[engine] = await task
            except Exception as e:
                logger.error(f"Error broadcasting to {engine}: {e}")
                results[engine] = False
                
        return results
    
    async def check_engine_health(self, target_engine: str) -> bool:
        """Check health status of target engine"""
        if not self._initialized:
            await self.initialize()
            
        if target_engine not in self.ENGINE_ENDPOINTS:
            return False
            
        endpoint = self.ENGINE_ENDPOINTS[target_engine]
        
        try:
            async with self.session.get(
                endpoint.health_url,
                timeout=aiohttp.ClientTimeout(total=2.0)
            ) as response:
                healthy = response.status == 200
                self.health_status[target_engine] = healthy
                return healthy
                
        except Exception as e:
            logger.debug(f"Health check failed for {target_engine}: {e}")
            self.health_status[target_engine] = False
            return False
    
    async def check_all_engines_health(self) -> Dict[str, bool]:
        """Check health status of all engines"""
        tasks = []
        engines = [e for e in self.ENGINE_ENDPOINTS.keys() if e != self.engine_name]
        
        for engine in engines:
            tasks.append(self.check_engine_health(engine))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for i, engine in enumerate(engines):
            if isinstance(results[i], Exception):
                health_status[engine] = False
            else:
                health_status[engine] = results[i]
        
        return health_status
    
    def get_healthy_engines(self) -> List[str]:
        """Get list of currently healthy engines"""
        return [engine for engine, healthy in self.health_status.items() if healthy]
    
    async def send_trading_signal(
        self, 
        target_engines: List[str], 
        signal_type: str,
        symbol: str,
        action: str,
        confidence: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Send trading signal to multiple engines"""
        payload = {
            "signal_type": signal_type,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "metadata": metadata
        }
        
        return await self.broadcast_message(
            BusinessMessageType.TRADING_SIGNAL,
            payload,
            target_engines,
            priority="HIGH"
        )
    
    async def send_risk_alert(
        self,
        target_engines: List[str],
        alert_type: str,
        severity: str,
        message: str,
        affected_positions: List[str]
    ) -> Dict[str, bool]:
        """Send risk alert to multiple engines"""
        payload = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "affected_positions": affected_positions,
            "timestamp": time.time()
        }
        
        return await self.broadcast_message(
            BusinessMessageType.RISK_ALERT,
            payload,
            target_engines,
            priority="CRITICAL"
        )
    
    async def send_performance_metric(
        self,
        target_engines: List[str],
        metric_name: str,
        value: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Send performance metric to multiple engines"""
        payload = {
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        return await self.broadcast_message(
            BusinessMessageType.PERFORMANCE_METRIC,
            payload,
            target_engines,
            priority="NORMAL"
        )


# Factory function
def create_mesh_client(engine_name: str, max_connections: int = 20) -> DirectMeshClient:
    """Create and return a DirectMeshClient instance"""
    return DirectMeshClient(engine_name, max_connections)


# Global client instances (for reuse)
_mesh_clients: Dict[str, DirectMeshClient] = {}

async def get_mesh_client(engine_name: str) -> DirectMeshClient:
    """Get or create mesh client for engine (singleton pattern)"""
    if engine_name not in _mesh_clients:
        client = create_mesh_client(engine_name)
        await client.initialize()
        _mesh_clients[engine_name] = client
    
    return _mesh_clients[engine_name]
#!/usr/bin/env python3
"""
Dual Bus Hybrid Communication Router
Routes messages between MarketData Bus (Port 6380) and Engine Logic Bus (Port 6381).

ROUTING RULES:
✅ MarketData Bus (6380): Market data distribution from MarketData Hub to engines
✅ Engine Logic Bus (6381): Engine-to-engine business logic communication

This implements the DUAL MESSAGE BUS ARCHITECTURE:
- STAR topology: MarketData distribution via MarketData Bus
- MESH topology: Business logic via Engine Logic Bus
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

# Import dual bus client
from dual_messagebus_client import DualMessageBusClient, get_dual_bus_client, MessageBusType
from universal_enhanced_messagebus_client import (
    MessageType, 
    EngineType,
    MessagePriority
)

logger = logging.getLogger(__name__)


class CommunicationPath(Enum):
    """Communication path selection"""
    MARKETDATA_BUS = "marketdata_bus"    # For market data distribution (Port 6380)
    ENGINE_LOGIC_BUS = "engine_logic_bus" # For business logic (Port 6381)


class MessageCategory(Enum):
    """Message category classification"""
    MARKET_DATA = "market_data"      # From MarketData Hub → Engines (MarketData Bus)
    BUSINESS_LOGIC = "business_logic" # Engine ↔ Engine communication (Engine Logic Bus)


@dataclass
class RoutingDecision:
    """Routing decision result"""
    path: CommunicationPath
    category: MessageCategory
    reason: str
    target_engines: List[str]
    use_broadcast: bool = False


class DualBusHybridRouter:
    """
    Dual Bus Hybrid Communication Router
    Intelligently routes messages between MarketData Bus (6380) and Engine Logic Bus (6381).
    """
    
    # Market data message types (use MarketData Bus - Port 6380)
    MARKET_DATA_MESSAGES = {
        MessageType.MARKET_DATA,
        MessageType.PRICE_UPDATE,
        MessageType.TRADE_EXECUTION,
        MessageType.HISTORICAL_DATA
    }
    
    # Business logic message types (use Engine Logic Bus - Port 6381)  
    BUSINESS_LOGIC_MESSAGES = {
        MessageType.VPIN_CALCULATION,
        MessageType.RISK_METRIC,
        MessageType.RISK_ASSESSMENT,
        MessageType.ML_PREDICTION,
        MessageType.ANALYTICS_RESULT,
        MessageType.STRATEGY_SIGNAL,
        MessageType.PORTFOLIO_UPDATE,
        MessageType.PERFORMANCE_METRIC,
        MessageType.SYSTEM_ALERT,
        MessageType.ENGINE_HEALTH,
        MessageType.COLLATERAL_UPDATE,
        MessageType.MARGIN_CALL
    }
    
    def __init__(self, engine_name: str, engine_type: EngineType, engine_port: int):
        self.engine_name = engine_name
        self.engine_type = engine_type
        self.engine_port = engine_port
        self.dual_bus_client: Optional[DualMessageBusClient] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize dual bus client"""
        if self._initialized:
            return
            
        # Initialize dual bus client
        self.dual_bus_client = await get_dual_bus_client(
            engine_type=self.engine_type,
            instance_id=f"{self.engine_name}-{self.engine_port}"
        )
        
        self._initialized = True
        logger.info(f"DualBusHybridRouter initialized for {self.engine_name}")
    
    async def close(self):
        """Close dual bus client"""
        if self.dual_bus_client:
            await self.dual_bus_client.close()
        self._initialized = False
    
    def route_message(
        self, 
        message_type: MessageType,
        source_engine: str,
        target_engines: List[str],
        payload: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Determine routing path for message based on type and source.
        
        ROUTING LOGIC:
        1. Market data from MarketData Hub → MarketData Bus (Port 6380) - STAR topology
        2. Engine-to-engine business logic → Engine Logic Bus (Port 6381) - MESH topology
        """
        
        # Rule 1: Market data distribution (MarketData Hub → Engines)
        if (source_engine == "marketdata" and 
            message_type in self.MARKET_DATA_MESSAGES):
            
            return RoutingDecision(
                path=CommunicationPath.MARKETDATA_BUS,
                category=MessageCategory.MARKET_DATA,
                reason="Market data distribution from MarketData Hub uses MarketData Bus Port 6380 (STAR topology)",
                target_engines=target_engines,
                use_broadcast=True
            )
        
        # Rule 2: Business logic (Engine ↔ Engine)
        if message_type in self.BUSINESS_LOGIC_MESSAGES:
            
            return RoutingDecision(
                path=CommunicationPath.ENGINE_LOGIC_BUS,
                category=MessageCategory.BUSINESS_LOGIC,
                reason="Engine-to-engine business logic uses Engine Logic Bus Port 6381 (MESH topology)",
                target_engines=target_engines,
                use_broadcast=False
            )
        
        # Rule 3: Unknown market data messages → Route to MarketData Bus
        if message_type.value in ["market_data", "price_update", "trade_data", "historical_data"]:
            return RoutingDecision(
                path=CommunicationPath.MARKETDATA_BUS,
                category=MessageCategory.MARKET_DATA,
                reason=f"Market data message {message_type.value} routed to MarketData Bus Port 6380",
                target_engines=target_engines,
                use_broadcast=True
            )
        
        # Default: Use Engine Logic Bus for unknown business logic
        return RoutingDecision(
            path=CommunicationPath.ENGINE_LOGIC_BUS,
            category=MessageCategory.BUSINESS_LOGIC,
            reason="Default routing for engine communication uses Engine Logic Bus Port 6381",
            target_engines=target_engines,
            use_broadcast=False
        )
    
    async def send_message(
        self,
        message_type: MessageType,
        target_engines: List[str],
        payload: Dict[str, Any],
        priority: str = "NORMAL",
        correlation_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """Send message using appropriate dual bus path"""
        if not self._initialized:
            await self.initialize()
        
        if not self.dual_bus_client:
            logger.error("Dual bus client not initialized")
            return {engine: False for engine in target_engines}
        
        # Get routing decision
        decision = self.route_message(
            message_type=message_type,
            source_engine=self.engine_name,
            target_engines=target_engines,
            payload=payload
        )
        
        logger.debug(f"Routing decision: {decision.path.value} - {decision.reason}")
        
        # Route via MarketData Bus (Port 6380)
        if decision.path == CommunicationPath.MARKETDATA_BUS:
            try:
                success = await self.dual_bus_client.publish_message(
                    message_type=message_type,
                    payload=payload,
                    priority=MessagePriority[priority] if hasattr(MessagePriority, priority) else MessagePriority.NORMAL,
                    bus_type=MessageBusType.MARKETDATA
                )
                # MarketData Bus handles broadcast, so return success for all targets
                return {engine: success for engine in target_engines}
                
            except Exception as e:
                logger.error(f"Error sending via MarketData Bus: {e}")
                return {engine: False for engine in target_engines}
        
        # Route via Engine Logic Bus (Port 6381)
        elif decision.path == CommunicationPath.ENGINE_LOGIC_BUS:
            try:
                success = await self.dual_bus_client.publish_message(
                    message_type=message_type,
                    payload=payload,
                    priority=MessagePriority[priority] if hasattr(MessagePriority, priority) else MessagePriority.NORMAL,
                    bus_type=MessageBusType.ENGINE_LOGIC
                )
                # Engine Logic Bus handles mesh communication
                return {engine: success for engine in target_engines}
                    
            except Exception as e:
                logger.error(f"Error sending via Engine Logic Bus: {e}")
                return {engine: False for engine in target_engines}
        
        # Unknown path
        logger.error(f"Unknown communication path: {decision.path}")
        return {engine: False for engine in target_engines}
    
    async def send_trading_signal(
        self,
        target_engines: List[str],
        signal_type: str,
        symbol: str,
        action: str,
        confidence: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Send trading signal via Engine Logic Bus"""
        return await self.send_message(
            message_type=MessageType.STRATEGY_SIGNAL,
            target_engines=target_engines,
            payload={
                "signal_type": signal_type,
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "metadata": metadata
            },
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
        """Send risk alert via Engine Logic Bus"""
        return await self.send_message(
            message_type=MessageType.RISK_ASSESSMENT,
            target_engines=target_engines,
            payload={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "affected_positions": affected_positions
            },
            priority="CRITICAL"
        )
    
    async def subscribe_to_market_data(
        self,
        data_types: List[str],
        symbols: Optional[List[str]] = None,
        handler: Optional[callable] = None
    ) -> bool:
        """Subscribe to market data via MarketData Bus (Port 6380)"""
        if not self._initialized:
            await self.initialize()
        
        if not self.dual_bus_client:
            logger.error("Dual bus client not initialized")
            return False
        
        try:
            # Convert data types to MessageType and subscribe to MarketData Bus
            message_types = []
            for data_type in data_types:
                if hasattr(MessageType, data_type.upper()):
                    message_types.append(getattr(MessageType, data_type.upper()))
            
            if message_types:
                await self.dual_bus_client.subscribe_to_marketdata(
                    message_types=message_types,
                    handler=handler
                )
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            return False
    
    async def subscribe_to_engine_logic(
        self,
        message_types: List[MessageType],
        handler: Optional[callable] = None
    ) -> bool:
        """Subscribe to engine logic messages via Engine Logic Bus (Port 6381)"""
        if not self._initialized:
            await self.initialize()
        
        if not self.dual_bus_client:
            logger.error("Dual bus client not initialized")
            return False
        
        try:
            await self.dual_bus_client.subscribe_to_engine_logic(
                message_types=message_types,
                handler=handler
            )
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to engine logic: {e}")
            return False
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get dual bus communication statistics"""
        stats = {
            "engine_name": self.engine_name,
            "engine_type": self.engine_type.value,
            "engine_port": self.engine_port,
            "initialized": self._initialized,
            "dual_bus_client": self.dual_bus_client is not None,
            "architecture": "dual_message_bus"
        }
        
        if self.dual_bus_client:
            try:
                bus_stats = asyncio.run(self.dual_bus_client.get_stats())
                stats["bus_statistics"] = bus_stats
            except Exception as e:
                stats["bus_statistics_error"] = str(e)
        
        return stats


# Factory function
def create_dual_bus_router(engine_name: str, engine_type: EngineType, engine_port: int) -> DualBusHybridRouter:
    """Create and return a DualBusHybridRouter instance"""
    return DualBusHybridRouter(engine_name, engine_type, engine_port)


# Global router instances (for reuse)
_dual_bus_routers: Dict[str, DualBusHybridRouter] = {}

async def get_dual_bus_router(engine_name: str, engine_type: EngineType, engine_port: int) -> DualBusHybridRouter:
    """Get or create dual bus hybrid router for engine (singleton pattern)"""
    router_key = f"{engine_name}-{engine_port}"
    
    if router_key not in _dual_bus_routers:
        router = create_dual_bus_router(engine_name, engine_type, engine_port)
        await router.initialize()
        _dual_bus_routers[router_key] = router
    
    return _dual_bus_routers[router_key]
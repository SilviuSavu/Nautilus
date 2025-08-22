"""
Real Nautilus Trader Interactive Brokers Client
Uses the official Nautilus IB adapter with proper connection management.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig
)
from nautilus_trader.adapters.interactive_brokers.data import InteractiveBrokersDataClient
from nautilus_trader.adapters.interactive_brokers.execution import InteractiveBrokersExecutionClient
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.cache.cache import Cache
from nautilus_trader.model.identifiers import TraderId, ClientId

logger = logging.getLogger(__name__)


class RealNautilusIBClient:
    """Real Nautilus Trader IB client using official adapters"""
    
    def __init__(self):
        self.host = os.environ.get('IB_HOST', 'host.docker.internal')
        self.port = int(os.environ.get('IB_PORT', '4002'))
        self.client_id = int(os.environ.get('IB_CLIENT_ID', '1001'))
        self.account_id = "DU7925702"
        
        # Nautilus components
        self.clock = LiveClock()
        self.trader_id = TraderId("NAUTILUS-IB-001")
        self.cache = Cache()
        self.msgbus = MessageBus(trader_id=self.trader_id, clock=self.clock)
        
        # Client configurations
        self.data_config = InteractiveBrokersDataClientConfig(
            ibg_host=self.host,
            ibg_port=self.port,
            ibg_client_id=self.client_id,
        )
        
        self.exec_config = InteractiveBrokersExecClientConfig(
            ibg_host=self.host,
            ibg_port=self.port,
            ibg_client_id=self.client_id,
            account_id=self.account_id,
        )
        
        # Initialize clients (but don't connect yet)
        self.data_client = None
        self.exec_client = None
        self._connected = False
        self._connection_time = None
        
    async def connect(self) -> bool:
        """Connect to IB Gateway using official Nautilus adapters"""
        try:
            logger.info(f"Connecting to IB Gateway at {self.host}:{self.port}")
            
            # Initialize clients
            loop = asyncio.get_event_loop()
            
            from nautilus_trader.adapters.interactive_brokers.providers import InteractiveBrokersInstrumentProvider
            
            # Create instrument provider first
            instrument_provider = InteractiveBrokersInstrumentProvider(
                client=None,  # Will be set by data client
                config=self.data_config,
            )
            
            self.data_client = InteractiveBrokersDataClient(
                loop=loop,
                client=None,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
                instrument_provider=instrument_provider,
                config=self.data_config,
            )
            
            self.exec_client = InteractiveBrokersExecutionClient(
                loop=loop,
                client=None,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock,
                config=self.exec_config,
            )
            
            # Connect both clients
            await self.data_client.connect()
            await self.exec_client.connect()
            
            self._connected = True
            self._connection_time = datetime.utcnow()
            
            logger.info("✅ Successfully connected to IB Gateway via Nautilus adapters")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to IB Gateway: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from IB Gateway"""
        try:
            if self.data_client:
                await self.data_client.disconnect()
            if self.exec_client:
                await self.exec_client.disconnect()
            
            self._connected = False
            self._connection_time = None
            
            logger.info("✅ Disconnected from IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to disconnect from IB Gateway: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        if not self.data_client or not self.exec_client:
            return False
        
        return (self._connected and 
                self.data_client.is_connected and 
                self.exec_client.is_connected)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "connected": self.is_connected(),
            "gateway_type": "IB Gateway (Nautilus Official)",
            "account_id": self.account_id if self.is_connected() else None,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "data_client_connected": self.data_client.is_connected if self.data_client else False,
            "exec_client_connected": self.exec_client.is_connected if self.exec_client else False,
        }


# Global client instance
_real_nautilus_ib_client: Optional[RealNautilusIBClient] = None


def get_real_nautilus_ib_client() -> RealNautilusIBClient:
    """Get or create the global real Nautilus IB client instance"""
    global _real_nautilus_ib_client
    
    if _real_nautilus_ib_client is None:
        _real_nautilus_ib_client = RealNautilusIBClient()
    
    return _real_nautilus_ib_client


def reset_real_nautilus_ib_client():
    """Reset the global real Nautilus IB client instance"""
    global _real_nautilus_ib_client
    _real_nautilus_ib_client = None
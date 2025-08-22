"""
Nautilus Trader Interactive Brokers Client Integration
Direct integration with the official Nautilus Trader IB adapter.
"""

import asyncio
import logging
from typing import Any, Optional
from datetime import datetime

from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig, InteractiveBrokersExecClientConfig
from nautilus_trader.adapters.interactive_brokers.data import InteractiveBrokersDataClient
from nautilus_trader.adapters.interactive_brokers.execution import InteractiveBrokersExecutionClient
from nautilus_trader.adapters.interactive_brokers.providers import InteractiveBrokersInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock, MessageBus
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import ClientId, TraderId


class NautilusIBClient:
    """
    Nautilus Trader Interactive Brokers client integration.
    """
    
    def __init__(
        self, 
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1001,
        account_id: str = "DU7925702"
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id
        
        # Nautilus components
        self.clock = LiveClock()
        self.trader_id = TraderId("NAUTILUS-001")
        self.cache = Cache()
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
        )
        
        # IB client configurations
        self.data_config = InteractiveBrokersDataClientConfig(
            ibg_host=host,
            ibg_port=port,
            ibg_client_id=client_id,
        )
        
        self.exec_config = InteractiveBrokersExecClientConfig(
            ibg_host=host,
            ibg_port=port,
            ibg_client_id=client_id,
            account_id=account_id,
        )
        
        # Initialize clients
        self.data_client = InteractiveBrokersDataClient(
            loop=asyncio.get_event_loop(),
            client=None,  # Will be set during connection
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
            config=self.data_config,
        )
        
        self.exec_client = InteractiveBrokersExecutionClient(
            loop=asyncio.get_event_loop(),
            client=None,  # Will be set during connection
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
            config=self.exec_config,
        )
        
        self.instrument_provider = self.data_client.instrument_provider
        
        self._connected = False
        self._connection_time = None
        
    async def connect(self) -> bool:
        """Connect to IB Gateway using Nautilus adapter."""
        try:
            await self.data_client.connect()
            await self.exec_client.connect()
            
            self._connected = True
            self._connection_time = datetime.utcnow()
            
            logging.info(f"Connected to IB Gateway via Nautilus adapter")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to IB Gateway: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from IB Gateway."""
        try:
            await self.data_client.disconnect()
            await self.exec_client.disconnect()
            
            self._connected = False
            self._connection_time = None
            
            logging.info("Disconnected from IB Gateway")
            return True
            
        except Exception as e:
            logging.error(f"Failed to disconnect from IB Gateway: {e}")
            return False
    
    @property
    def connected(self) -> bool:
        """Check if connected to IB Gateway."""
        return self._connected and self.data_client.is_connected and self.exec_client.is_connected
    
    @property
    def connection_info(self) -> dict:
        """Get connection information."""
        return {
            "connected": self.connected,
            "gateway_type": "IB Gateway (Nautilus)",
            "account_id": self.account_id if self.connected else None,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "next_valid_order_id": 1,  # Managed by Nautilus
            "server_version": 0,  # Not exposed by Nautilus
            "error_message": None,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
        }


# Global client instance
_nautilus_ib_client: Optional[NautilusIBClient] = None


def get_nautilus_ib_client(
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 1001,
    account_id: str = "DU7925702"
) -> NautilusIBClient:
    """Get or create the global Nautilus IB client instance."""
    global _nautilus_ib_client
    
    if _nautilus_ib_client is None:
        _nautilus_ib_client = NautilusIBClient(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id
        )
    
    return _nautilus_ib_client


def reset_nautilus_ib_client():
    """Reset the global Nautilus IB client instance."""
    global _nautilus_ib_client
    _nautilus_ib_client = None
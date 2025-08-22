"""
Nautilus Trader TradingNode Integration
Manages the official Nautilus Trader node with Interactive Brokers adapter.
"""

import asyncio
import logging
import os
from typing import Optional
from datetime import datetime

from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.config import IBMarketDataTypeEnum
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersExecClientConfig
from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersInstrumentProviderConfig
from nautilus_trader.adapters.interactive_brokers.config import SymbologyMethod
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveDataClientFactory
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveExecClientFactory
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import RoutingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId


class NautilusTradingNodeManager:
    """
    Manages the Nautilus Trader TradingNode for IB integration.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = None,
        account_id: str = "DU7925702",
        trader_id: str = "NAUTILUS-001"
    ):
        # Use environment variables with fallbacks
        self.host = host or os.environ.get('IB_HOST', 'host.docker.internal')
        self.port = port or int(os.environ.get('IB_PORT', '4002'))
        self.client_id = client_id or int(os.environ.get('IB_CLIENT_ID', '1001'))
        self.account_id = account_id
        self.trader_id = trader_id
        
        self._node: Optional[TradingNode] = None
        self._connected = False
        self._connection_time = None
        
        # Configure instrument provider with common instruments
        self.instrument_provider_config = InteractiveBrokersInstrumentProviderConfig(
            symbology_method=SymbologyMethod.IB_SIMPLIFIED,
            load_ids=frozenset([
                "EUR/USD.IDEALPRO",
                "GBP/USD.IDEALPRO", 
                "USD/JPY.IDEALPRO",
                "AAPL.NASDAQ",
                "MSFT.NASDAQ",
                "GOOGL.NASDAQ",
                "TSLA.NASDAQ",
                "SPY.ARCA",
                "QQQ.NASDAQ",
            ]),
        )
        
        # Configure TradingNode
        self.config = TradingNodeConfig(
            trader_id=trader_id,
            logging=LoggingConfig(log_level="INFO"),
            data_clients={
                IB: InteractiveBrokersDataClientConfig(
                    ibg_host=self.host,
                    ibg_port=self.port,
                    ibg_client_id=self.client_id,
                    handle_revised_bars=False,
                    use_regular_trading_hours=True,
                    market_data_type=IBMarketDataTypeEnum.REALTIME,
                    instrument_provider=self.instrument_provider_config,
                ),
            },
            exec_clients={
                IB: InteractiveBrokersExecClientConfig(
                    ibg_host=self.host,
                    ibg_port=self.port,
                    ibg_client_id=self.client_id,
                    account_id=self.account_id,
                    instrument_provider=self.instrument_provider_config,
                    routing=RoutingConfig(default=True),
                ),
            },
            data_engine=LiveDataEngineConfig(
                time_bars_timestamp_on_close=False,
                validate_data_sequence=True,
            ),
            timeout_connection=90.0,
            timeout_reconciliation=5.0,
            timeout_portfolio=5.0,
            timeout_disconnection=5.0,
            timeout_post_stop=2.0,
        )
        
        logging.info(f"Initialized Nautilus TradingNode Manager for {trader_id}")
    
    async def start(self) -> bool:
        """Start the Nautilus TradingNode."""
        try:
            if self._node is not None:
                logging.warning("TradingNode already started")
                # Update connection state even if already started
                self._connected = True
                if not self._connection_time:
                    self._connection_time = datetime.utcnow()
                return True
                
            logging.info("Starting Nautilus TradingNode...")
            
            # Create and build the node
            self._node = TradingNode(config=self.config)
            
            # Register Interactive Brokers factories
            self._node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
            self._node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
            
            # Build the node (this initializes all components)
            await self._node.build()
            
            # Start the node
            await self._node.start()
            
            self._connected = True
            self._connection_time = datetime.utcnow()
            
            logging.info("✅ Nautilus TradingNode started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start Nautilus TradingNode: {e}")
            self._connected = False
            return False
    
    async def stop(self) -> bool:
        """Stop the Nautilus TradingNode."""
        try:
            if self._node is None:
                logging.warning("TradingNode not started")
                return True
                
            logging.info("Stopping Nautilus TradingNode...")
            
            # Stop the node
            await self._node.stop()
            
            self._node = None
            self._connected = False
            self._connection_time = None
            
            logging.info("✅ Nautilus TradingNode stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop Nautilus TradingNode: {e}")
            return False
    
    @property
    def connected(self) -> bool:
        """Check if the TradingNode is connected and running."""
        return self._connected and self._node is not None
    
    @property
    def node(self) -> Optional[TradingNode]:
        """Get the TradingNode instance."""
        return self._node
    
    @property
    def connection_info(self) -> dict:
        """Get connection information."""
        return {
            "connected": self.connected,
            "gateway_type": "IB Gateway (Nautilus Trader)",
            "account_id": self.account_id if self.connected else None,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "next_valid_order_id": 1,  # Managed internally by Nautilus
            "server_version": 0,  # Not exposed
            "error_message": None,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "trader_id": self.trader_id,
        }
    
    def get_message_bus(self):
        """Get the message bus from the TradingNode."""
        if self._node is None:
            return None
        return self._node.kernel.msgbus
    
    def get_cache(self):
        """Get the cache from the TradingNode."""
        if self._node is None:
            return None
        return self._node.kernel.cache
    
    def get_portfolio(self):
        """Get the portfolio from the TradingNode.""" 
        if self._node is None:
            return None
        return self._node.kernel.portfolio


# Global instance
_nautilus_node_manager: Optional[NautilusTradingNodeManager] = None


def get_nautilus_node_manager(
    host: str = None,
    port: int = None,
    client_id: int = None,
    account_id: str = "DU7925702"
) -> NautilusTradingNodeManager:
    """Get or create the global Nautilus TradingNode manager."""
    global _nautilus_node_manager
    
    if _nautilus_node_manager is None:
        _nautilus_node_manager = NautilusTradingNodeManager(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id
        )
    
    return _nautilus_node_manager


def reset_nautilus_node_manager():
    """Reset the global Nautilus TradingNode manager."""
    global _nautilus_node_manager
    _nautilus_node_manager = None
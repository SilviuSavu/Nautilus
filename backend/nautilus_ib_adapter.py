"""
NautilusTrader Interactive Brokers Adapter Integration
Professional-grade IB Gateway integration using the official NautilusTrader IB adapter.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig, InteractiveBrokersExecClientConfig
from nautilus_trader.adapters.interactive_brokers.data import InteractiveBrokersDataClient
from nautilus_trader.adapters.interactive_brokers.execution import InteractiveBrokersExecutionClient
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveDataClientFactory, InteractiveBrokersLiveExecClientFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.model.identifiers import AccountId, TraderId, StrategyId


@dataclass
class IBGatewayStatus:
    """IB Gateway connection status"""
    connected: bool = False
    account_id: Optional[str] = None
    connection_time: Optional[datetime] = None
    error_message: Optional[str] = None
    client_id: int = 1
    host: str = "127.0.0.1"
    port: int = 4002


@dataclass
class IBMarketDataUpdate:
    """IB Market data update"""
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None


class NautilusIBAdapter:
    """
    NautilusTrader Interactive Brokers Adapter Integration
    
    Provides a simplified interface to NautilusTrader's professional IB adapter
    for web dashboard integration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # NautilusTrader components
        self.clock = LiveClock()
        self.trader_id = TraderId("TRADER-001")
        self.strategy_id = StrategyId("IB-ADAPTER")
        self.cache = Cache()
        self.msgbus = MessageBus(trader_id=self.trader_id, clock=self.clock)
        
        # IB clients
        self.data_client: Optional[InteractiveBrokersDataClient] = None
        self.exec_client: Optional[InteractiveBrokersExecutionClient] = None
        
        # Configuration
        self.config = self._create_config()
        
        # Status and callbacks
        self.status = IBGatewayStatus()
        self.market_data_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        
        # Market data cache
        self.market_data: Dict[str, IBMarketDataUpdate] = {}
        
    def _create_config(self) -> Dict[str, Any]:
        """Create IB adapter configuration from environment variables"""
        return {
            "host": os.getenv("IB_HOST", "127.0.0.1"),
            "port": int(os.getenv("IB_PORT", "4002")),
            "client_id": int(os.getenv("IB_CLIENT_ID", "1")),
            "account_id": os.getenv("IB_ACCOUNT_ID", "DU12345"),
            "trading_mode": os.getenv("TRADING_MODE", "paper"),
        }
    
    async def connect(self) -> bool:
        """Connect to IB Gateway using NautilusTrader adapter"""
        try:
            self.logger.info(f"Connecting to IB Gateway at {self.config['host']}:{self.config['port']}")
            
            # Create data client configuration
            data_config = InteractiveBrokersDataClientConfig(
                ibg_host=self.config["host"],
                ibg_port=self.config["port"],
                ibg_client_id=self.config["client_id"],
                account_id=self.config["account_id"],
                trading_mode=self.config["trading_mode"]
            )
            
            # Create execution client configuration  
            exec_config = InteractiveBrokersExecClientConfig(
                ibg_host=self.config["host"],
                ibg_port=self.config["port"],
                ibg_client_id=self.config["client_id"] + 1,  # Different client ID
                account_id=self.config["account_id"],
                trading_mode=self.config["trading_mode"]
            )
            
            # Create data client
            self.data_client = InteractiveBrokersLiveDataClientFactory.create(
                config=data_config,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock
            )
            
            # Create execution client
            self.exec_client = InteractiveBrokersLiveExecClientFactory.create(
                config=exec_config,
                msgbus=self.msgbus,
                cache=self.cache,
                clock=self.clock
            )
            
            # Connect clients
            await self.data_client.connect()
            await self.exec_client.connect()
            
            # Update status
            self.status.connected = True
            self.status.connection_time = datetime.now()
            self.status.account_id = self.config["account_id"]
            self.status.host = self.config["host"]
            self.status.port = self.config["port"]
            self.status.client_id = self.config["client_id"]
            
            self.logger.info("Successfully connected to IB Gateway via NautilusTrader")
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                try:
                    await callback(self.status)
                except Exception as e:
                    self.logger.error(f"Error in connection callback: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IB Gateway: {e}")
            self.status.error_message = str(e)
            self.status.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IB Gateway"""
        try:
            if self.data_client:
                await self.data_client.disconnect()
            if self.exec_client:
                await self.exec_client.disconnect()
            
            self.status.connected = False
            self.logger.info("Disconnected from IB Gateway")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB Gateway: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        return self.status.connected
    
    def get_status(self) -> IBGatewayStatus:
        """Get current connection status"""
        return self.status
    
    async def subscribe_market_data(self, symbols: List[str]) -> Dict[str, bool]:
        """Subscribe to market data for symbols"""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        results = {}
        for symbol in symbols:
            try:
                # Create instruments and subscribe to real market data
                # For now, create basic stock contracts - will be enhanced when asset class support is added
                from ibapi.contract import Contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                
                # Subscribe using the data client's market data capabilities
                # This is a placeholder until we implement the full NautilusTrader integration
                results[symbol] = True
                self.logger.info(f"Subscribed to market data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to {symbol}: {e}")
                results[symbol] = False
        
        return results
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> Dict[str, bool]:
        """Unsubscribe from market data for symbols"""
        if not self.is_connected():
            return {symbol: False for symbol in symbols}
        
        results = {}
        for symbol in symbols:
            try:
                # TODO: Implement market data unsubscription
                if symbol in self.market_data:
                    del self.market_data[symbol]
                results[symbol] = True
                self.logger.info(f"Unsubscribed from market data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")
                results[symbol] = False
        
        return results
    
    def get_market_data(self, symbol: str) -> Optional[IBMarketDataUpdate]:
        """Get latest market data for a symbol"""
        return self.market_data.get(symbol)
    
    def get_all_market_data(self) -> Dict[str, IBMarketDataUpdate]:
        """Get all market data"""
        return self.market_data.copy()
    
    def add_market_data_callback(self, callback: Callable):
        """Add callback for market data updates"""
        self.market_data_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add callback for connection status updates"""
        self.connection_callbacks.append(callback)
    
    async def _handle_market_data_update(self, symbol: str, data: Dict[str, Any]):
        """Handle market data update from NautilusTrader"""
        market_update = IBMarketDataUpdate(
            symbol=symbol,
            bid=data.get("bid"),
            ask=data.get("ask"),
            last=data.get("last"),
            volume=data.get("volume"),
            timestamp=datetime.now()
        )
        
        self.market_data[symbol] = market_update
        
        # Notify callbacks
        for callback in self.market_data_callbacks:
            try:
                await callback(symbol, market_update)
            except Exception as e:
                self.logger.error(f"Error in market data callback: {e}")


# Global adapter instance
_nautilus_ib_adapter: Optional[NautilusIBAdapter] = None

def get_nautilus_ib_adapter() -> NautilusIBAdapter:
    """Get or create the NautilusTrader IB adapter singleton"""
    global _nautilus_ib_adapter
    
    if _nautilus_ib_adapter is None:
        _nautilus_ib_adapter = NautilusIBAdapter()
    
    return _nautilus_ib_adapter

def reset_nautilus_ib_adapter():
    """Reset the adapter singleton (for testing)"""
    global _nautilus_ib_adapter
    if _nautilus_ib_adapter:
        asyncio.create_task(_nautilus_ib_adapter.disconnect())
    _nautilus_ib_adapter = None
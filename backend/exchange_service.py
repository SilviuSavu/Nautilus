"""
Exchange Adapter Service
Provides unified interface for connecting to multiple cryptocurrency exchanges
with secure credential management and live trading capabilities.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json

from enums import Venue


class ExchangeStatus(Enum):
    """Exchange connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class TradingMode(Enum):
    """Trading modes"""
    PAPER = "paper"          # Paper trading (simulation)
    TESTNET = "testnet"      # Exchange testnet
    LIVE = "live"            # Live trading


@dataclass
class ExchangeCredentials:
    """Exchange API credentials"""
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # For some exchanges like Coinbase
    sandbox: bool = True  # Default to sandbox/testnet
    base_url: Optional[str] = None  # Custom base URL for testnet
    ws_url: Optional[str] = None    # Custom WebSocket URL for testnet
    
    
@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    venue: Venue
    credentials: Optional[ExchangeCredentials] = None
    trading_mode: TradingMode = TradingMode.PAPER
    enabled: bool = False
    max_orders_per_minute: int = 100
    position_size_limit: float = 10000.0  # USD equivalent
    

@dataclass
class ExchangeConnection:
    """Exchange connection information"""
    venue: Venue
    status: ExchangeStatus
    config: ExchangeConfig
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    supported_features: List[str] = None


class ExchangeService:
    """
    Service for managing connections to cryptocurrency exchanges
    with secure credential management and live trading capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._connections: Dict[Venue, ExchangeConnection] = {}
        self._configs: Dict[Venue, ExchangeConfig] = {}
        self._load_configurations()
        
    def _load_configurations(self) -> None:
        """Load exchange configurations from environment variables"""
        # Define supported exchanges with their environment variable patterns
        exchanges = {
            Venue.BINANCE: "BINANCE",
            Venue.COINBASE: "COINBASE", 
            Venue.KRAKEN: "KRAKEN",
            Venue.BYBIT: "BYBIT",
            Venue.OKX: "OKX",
        }
        
        for venue, env_prefix in exchanges.items():
            config = self._load_exchange_config(venue, env_prefix)
            if config:
                self._configs[venue] = config
                self._connections[venue] = ExchangeConnection(
                    venue=venue,
                    status=ExchangeStatus.DISCONNECTED,
                    config=config,
                    supported_features=self._get_supported_features(venue)
                )
                
    def _load_exchange_config(self, venue: Venue, env_prefix: str) -> Optional[ExchangeConfig]:
        """Load configuration for a specific exchange"""
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        api_secret = os.getenv(f"{env_prefix}_API_SECRET")
        
        if not api_key or not api_secret:
            self.logger.info(f"No credentials found for {venue.value}")
            return ExchangeConfig(
                venue=venue,
                enabled=False,
                trading_mode=TradingMode.PAPER
            )
            
        passphrase = os.getenv(f"{env_prefix}_PASSPHRASE")  # For Coinbase
        sandbox = os.getenv(f"{env_prefix}_SANDBOX", "true").lower() == "true"
        trading_mode_str = os.getenv(f"{env_prefix}_TRADING_MODE", "paper").lower()
        base_url = os.getenv(f"{env_prefix}_BASE_URL")  # Custom base URL for testnet
        ws_url = os.getenv(f"{env_prefix}_WS_URL")      # Custom WebSocket URL for testnet
        
        try:
            trading_mode = TradingMode(trading_mode_str)
        except ValueError:
            trading_mode = TradingMode.PAPER
            
        credentials = ExchangeCredentials(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            sandbox=sandbox,
            base_url=base_url,
            ws_url=ws_url
        )
        
        return ExchangeConfig(
            venue=venue,
            credentials=credentials,
            trading_mode=trading_mode,
            enabled=True,
            max_orders_per_minute=int(os.getenv(f"{env_prefix}_MAX_ORDERS_PER_MINUTE", "100")),
            position_size_limit=float(os.getenv(f"{env_prefix}_POSITION_LIMIT", "10000"))
        )
        
    def _get_supported_features(self, venue: Venue) -> List[str]:
        """Get supported features for an exchange"""
        feature_map = {
            Venue.BINANCE: ["spot_trading", "futures_trading", "margin_trading", "websocket_streams"],
            Venue.COINBASE: ["spot_trading", "websocket_streams", "advanced_orders"],
            Venue.KRAKEN: ["spot_trading", "futures_trading", "websocket_streams"],
            Venue.BYBIT: ["spot_trading", "derivatives_trading", "websocket_streams"],
            Venue.OKX: ["spot_trading", "futures_trading", "options_trading", "websocket_streams"],
        }
        return feature_map.get(venue, ["spot_trading"])
        
    async def connect_exchange(self, venue: Venue) -> bool:
        """Connect to a specific exchange"""
        if venue not in self._connections:
            self.logger.error(f"Exchange {venue.value} not configured")
            return False
            
        connection = self._connections[venue]
        config = connection.config
        
        if not config.enabled or not config.credentials:
            self.logger.warning(f"Exchange {venue.value} not enabled or missing credentials")
            return False
            
        try:
            connection.status = ExchangeStatus.CONNECTING
            self.logger.info(f"Connecting to {venue.value} in {config.trading_mode.value} mode")
            
            # TODO: Initialize actual exchange adapters here
            # For now, simulate connection based on credential presence
            if config.credentials and config.credentials.api_key:
                await asyncio.sleep(1)  # Simulate connection time
                connection.status = ExchangeStatus.CONNECTED
                connection.last_heartbeat = datetime.now()
                connection.error_message = None
                self.logger.info(f"Connected to {venue.value}")
                return True
            else:
                raise ValueError("Invalid credentials")
                
        except Exception as e:
            connection.status = ExchangeStatus.ERROR
            connection.error_message = str(e)
            self.logger.error(f"Failed to connect to {venue.value}: {e}")
            return False
            
    async def disconnect_exchange(self, venue: Venue) -> None:
        """Disconnect from a specific exchange"""
        if venue in self._connections:
            connection = self._connections[venue]
            connection.status = ExchangeStatus.DISCONNECTED
            connection.last_heartbeat = None
            self.logger.info(f"Disconnected from {venue.value}")
            
    async def connect_all_exchanges(self) -> Dict[Venue, bool]:
        """Connect to all configured exchanges"""
        results = {}
        for venue in self._connections.keys():
            results[venue] = await self.connect_exchange(venue)
        return results
        
    async def disconnect_all_exchanges(self) -> None:
        """Disconnect from all exchanges"""
        for venue in list(self._connections.keys()):
            await self.disconnect_exchange(venue)
            
    def get_exchange_status(self, venue: Venue) -> Optional[ExchangeConnection]:
        """Get status for a specific exchange"""
        return self._connections.get(venue)
        
    def get_all_exchange_status(self) -> Dict[Venue, ExchangeConnection]:
        """Get status for all exchanges"""
        return self._connections.copy()
        
    def is_exchange_connected(self, venue: Venue) -> bool:
        """Check if exchange is connected"""
        connection = self._connections.get(venue)
        return connection and connection.status == ExchangeStatus.CONNECTED
        
    def get_connected_exchanges(self) -> List[Venue]:
        """Get list of connected exchanges"""
        return [
            venue for venue, connection in self._connections.items()
            if connection.status == ExchangeStatus.CONNECTED
        ]
        
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading system summary"""
        total_exchanges = len(self._connections)
        connected_exchanges = len(self.get_connected_exchanges())
        enabled_exchanges = len([c for c in self._connections.values() if c.config.enabled])
        
        trading_modes = {}
        for connection in self._connections.values():
            mode = connection.config.trading_mode.value
            trading_modes[mode] = trading_modes.get(mode, 0) + 1
            
        return {
            "total_exchanges": total_exchanges,
            "enabled_exchanges": enabled_exchanges,
            "connected_exchanges": connected_exchanges,
            "trading_modes": trading_modes,
            "supported_venues": [venue.value for venue in self._connections.keys()]
        }
        
    async def health_check(self) -> None:
        """Perform health check on all exchange connections"""
        for venue, connection in self._connections.items():
            if connection.status == ExchangeStatus.CONNECTED:
                # Check if connection is stale
                if (connection.last_heartbeat and 
                    (datetime.now() - connection.last_heartbeat).seconds > 300):  # 5 minutes
                    self.logger.warning(f"Exchange {venue.value} connection appears stale")
                    connection.status = ExchangeStatus.ERROR
                    connection.error_message = "Connection timeout"


# Global exchange service instance
exchange_service = ExchangeService()
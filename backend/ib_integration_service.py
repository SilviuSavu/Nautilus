"""
Interactive Brokers Integration Service
Provides specialized handling for IB data streams, account monitoring, and order routing.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from decimal import Decimal

from messagebus_client import MessageBusClient, MessageBusMessage
from market_data_service import MarketDataService, NormalizedMarketData
from enums import Venue, DataType, MessageBusTopics


@dataclass
class IBConnectionStatus:
    """Interactive Brokers connection status"""
    connected: bool = False
    gateway_type: str = "TWS"  # TWS or IBG
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1
    account_id: Optional[str] = None
    connection_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class IBAccountData:
    """Interactive Brokers account information"""
    account_id: str
    net_liquidation: Optional[Decimal] = None
    total_cash_value: Optional[Decimal] = None
    buying_power: Optional[Decimal] = None
    maintenance_margin: Optional[Decimal] = None
    initial_margin: Optional[Decimal] = None
    excess_liquidity: Optional[Decimal] = None
    currency: str = "USD"
    timestamp: Optional[datetime] = None


@dataclass
class IBPosition:
    """Interactive Brokers position data"""
    account_id: str
    contract_id: str
    symbol: str
    position: Decimal
    avg_cost: Optional[Decimal] = None
    market_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    timestamp: Optional[datetime] = None


@dataclass
class IBOrderData:
    """Interactive Brokers order information"""
    order_id: str
    client_id: int
    account_id: str
    contract_id: str
    symbol: str
    action: str  # BUY/SELL
    order_type: str  # MKT, LMT, STP, etc.
    total_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: str = "PendingSubmit"
    avg_fill_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    timestamp: Optional[datetime] = None


class IBIntegrationService:
    """
    Interactive Brokers Integration Service
    
    Handles IB-specific MessageBus subscriptions, data parsing, and
    account/order management for the NautilusTrader web dashboard.
    """
    
    def __init__(self, messagebus_client: MessageBusClient):
        self.logger = logging.getLogger(__name__)
        self.messagebus_client = messagebus_client
        
        # IB-specific state
        self.connection_status = IBConnectionStatus()
        self.account_data: Optional[IBAccountData] = None
        self.positions: Dict[str, IBPosition] = {}
        self.orders: Dict[str, IBOrderData] = {}
        
        # Event handlers
        self._account_handlers: List[Callable[[IBAccountData], None]] = []
        self._position_handlers: List[Callable[[Dict[str, IBPosition]], None]] = []
        self._order_handlers: List[Callable[[IBOrderData], None]] = []
        self._connection_handlers: List[Callable[[IBConnectionStatus], None]] = []
        
        # Setup MessageBus handlers
        self._setup_messagebus_handlers()
    
    def _setup_messagebus_handlers(self):
        """Setup MessageBus topic subscriptions for IB data"""
        # Subscribe to IB-specific topics via a unified handler
        self.messagebus_client.add_message_handler(self._handle_messagebus_message)
    
    async def _handle_messagebus_message(self, message: MessageBusMessage):
        """Unified MessageBus message handler that routes to specific handlers"""
        topic = message.topic
        
        try:
            if topic == "adapter.interactive_brokers.account":
                await self._handle_account_update(message)
            elif topic == "adapter.interactive_brokers.position":
                await self._handle_position_update(message)
            elif topic == "adapter.interactive_brokers.order":
                await self._handle_order_update(message)
            elif topic == "adapter.interactive_brokers.connection":
                await self._handle_connection_update(message)
            elif topic in ["data.quotes.IB", "data.trades.IB", "data.bars.IB"]:
                await self._handle_market_data(message)
            else:
                # Ignore non-IB related messages
                pass
        except Exception as e:
            self.logger.error(f"Error handling MessageBus message {topic}: {e}")
    
    async def _handle_connection_update(self, message: MessageBusMessage):
        """Handle IB connection status updates"""
        try:
            payload = message.payload
            
            self.connection_status.connected = payload.get("connected", False)
            self.connection_status.gateway_type = payload.get("gateway_type", "TWS")
            self.connection_status.host = payload.get("host", "127.0.0.1")
            self.connection_status.port = payload.get("port", 4002)
            self.connection_status.client_id = payload.get("client_id", 1)
            self.connection_status.account_id = payload.get("account_id")
            
            if payload.get("connected"):
                self.connection_status.connection_time = datetime.now()
            
            if payload.get("error"):
                self.connection_status.error_message = payload.get("error")
            
            self.connection_status.last_heartbeat = datetime.now()
            
            # Notify handlers
            for handler in self._connection_handlers:
                try:
                    await handler(self.connection_status)
                except Exception as e:
                    self.logger.error(f"Error in connection handler: {e}")
                    
            self.logger.info(f"IB Connection status updated: {self.connection_status.connected}")
            
        except Exception as e:
            self.logger.error(f"Error handling IB connection update: {e}")
    
    async def _handle_account_update(self, message: MessageBusMessage):
        """Handle IB account data updates"""
        try:
            payload = message.payload
            account_id = payload.get("account_id")
            
            if not account_id:
                return
            
            # Update or create account data
            if not self.account_data or self.account_data.account_id != account_id:
                self.account_data = IBAccountData(account_id=account_id)
            
            # Update account values
            account_values = payload.get("account_values", {})
            
            if "NetLiquidation" in account_values:
                self.account_data.net_liquidation = Decimal(str(account_values["NetLiquidation"]))
            if "TotalCashValue" in account_values:
                self.account_data.total_cash_value = Decimal(str(account_values["TotalCashValue"]))
            if "BuyingPower" in account_values:
                self.account_data.buying_power = Decimal(str(account_values["BuyingPower"]))
            if "MaintMarginReq" in account_values:
                self.account_data.maintenance_margin = Decimal(str(account_values["MaintMarginReq"]))
            if "InitMarginReq" in account_values:
                self.account_data.initial_margin = Decimal(str(account_values["InitMarginReq"]))
            if "ExcessLiquidity" in account_values:
                self.account_data.excess_liquidity = Decimal(str(account_values["ExcessLiquidity"]))
            
            self.account_data.currency = payload.get("currency", "USD")
            self.account_data.timestamp = datetime.now()
            
            # Notify handlers
            for handler in self._account_handlers:
                try:
                    await handler(self.account_data)
                except Exception as e:
                    self.logger.error(f"Error in account handler: {e}")
                    
            self.logger.debug(f"IB Account updated: {account_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling IB account update: {e}")
    
    async def _handle_position_update(self, message: MessageBusMessage):
        """Handle IB position updates"""
        try:
            payload = message.payload
            account_id = payload.get("account_id")
            contract_id = payload.get("contract_id")
            
            if not account_id or not contract_id:
                return
            
            position_key = f"{account_id}:{contract_id}"
            
            # Create or update position
            if position_key not in self.positions:
                self.positions[position_key] = IBPosition(
                    account_id=account_id,
                    contract_id=contract_id,
                    symbol=payload.get("symbol", ""),
                    position=Decimal("0")
                )
            
            position = self.positions[position_key]
            
            # Update position data
            if "position" in payload:
                position.position = Decimal(str(payload["position"]))
            if "avg_cost" in payload:
                position.avg_cost = Decimal(str(payload["avg_cost"]))
            if "market_price" in payload:
                position.market_price = Decimal(str(payload["market_price"]))
            if "market_value" in payload:
                position.market_value = Decimal(str(payload["market_value"]))
            if "unrealized_pnl" in payload:
                position.unrealized_pnl = Decimal(str(payload["unrealized_pnl"]))
            if "realized_pnl" in payload:
                position.realized_pnl = Decimal(str(payload["realized_pnl"]))
            
            position.timestamp = datetime.now()
            
            # Notify handlers
            for handler in self._position_handlers:
                try:
                    await handler(self.positions)
                except Exception as e:
                    self.logger.error(f"Error in position handler: {e}")
                    
            self.logger.debug(f"IB Position updated: {position_key}")
            
        except Exception as e:
            self.logger.error(f"Error handling IB position update: {e}")
    
    async def _handle_order_update(self, message: MessageBusMessage):
        """Handle IB order updates"""
        try:
            payload = message.payload
            order_id = payload.get("order_id")
            
            if not order_id:
                return
            
            # Create or update order
            if order_id not in self.orders:
                self.orders[order_id] = IBOrderData(
                    order_id=order_id,
                    client_id=payload.get("client_id", 0),
                    account_id=payload.get("account_id", ""),
                    contract_id=payload.get("contract_id", ""),
                    symbol=payload.get("symbol", ""),
                    action=payload.get("action", ""),
                    order_type=payload.get("order_type", ""),
                    total_quantity=Decimal(str(payload.get("total_quantity", 0))),
                    filled_quantity=Decimal("0"),
                    remaining_quantity=Decimal(str(payload.get("total_quantity", 0)))
                )
            
            order = self.orders[order_id]
            
            # Update order data
            if "filled_quantity" in payload:
                order.filled_quantity = Decimal(str(payload["filled_quantity"]))
                order.remaining_quantity = order.total_quantity - order.filled_quantity
            
            if "limit_price" in payload:
                order.limit_price = Decimal(str(payload["limit_price"]))
            if "stop_price" in payload:
                order.stop_price = Decimal(str(payload["stop_price"]))
            if "status" in payload:
                order.status = payload["status"]
            if "avg_fill_price" in payload:
                order.avg_fill_price = Decimal(str(payload["avg_fill_price"]))
            if "commission" in payload:
                order.commission = Decimal(str(payload["commission"]))
            
            order.timestamp = datetime.now()
            
            # Notify handlers
            for handler in self._order_handlers:
                try:
                    await handler(order)
                except Exception as e:
                    self.logger.error(f"Error in order handler: {e}")
                    
            self.logger.debug(f"IB Order updated: {order_id} - {order.status}")
            
        except Exception as e:
            self.logger.error(f"Error handling IB order update: {e}")
    
    async def _handle_market_data(self, message: MessageBusMessage):
        """Handle IB market data updates"""
        try:
            payload = message.payload
            
            # Create normalized market data for IB
            normalized_data = NormalizedMarketData(
                venue="IB",
                instrument_id=payload.get("instrument_id", ""),
                data_type=payload.get("data_type", ""),
                timestamp=message.timestamp,
                data=payload.get("data", {}),
                raw_data=payload
            )
            
            self.logger.debug(f"IB Market data: {normalized_data.instrument_id} - {normalized_data.data_type}")
            
        except Exception as e:
            self.logger.error(f"Error handling IB market data: {e}")
    
    # Event handler registration methods
    def add_account_handler(self, handler: Callable[[IBAccountData], None]):
        """Add handler for account updates"""
        self._account_handlers.append(handler)
    
    def add_position_handler(self, handler: Callable[[Dict[str, IBPosition]], None]):
        """Add handler for position updates"""
        self._position_handlers.append(handler)
    
    def add_order_handler(self, handler: Callable[[IBOrderData], None]):
        """Add handler for order updates"""
        self._order_handlers.append(handler)
    
    def add_connection_handler(self, handler: Callable[[IBConnectionStatus], None]):
        """Add handler for connection status updates"""
        self._connection_handlers.append(handler)
    
    # Public API methods
    async def get_connection_status(self) -> IBConnectionStatus:
        """Get current IB connection status"""
        return self.connection_status
    
    async def get_account_data(self) -> Optional[IBAccountData]:
        """Get current account data"""
        return self.account_data
    
    async def get_positions(self) -> Dict[str, IBPosition]:
        """Get all current positions"""
        return self.positions.copy()
    
    async def get_orders(self) -> Dict[str, IBOrderData]:
        """Get all orders"""
        return self.orders.copy()
    
    async def request_account_summary(self, account_id: str):
        """Request account summary update"""
        message = MessageBusMessage(
            topic="request.ib.account_summary",
            payload={"account_id": account_id},
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)
    
    async def request_positions(self, account_id: str):
        """Request positions update"""
        message = MessageBusMessage(
            topic="request.ib.positions",
            payload={"account_id": account_id},
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)
    
    async def request_open_orders(self, account_id: str):
        """Request open orders update"""
        message = MessageBusMessage(
            topic="request.ib.open_orders",
            payload={"account_id": account_id},
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)
    
    async def place_order(self, order_request: Dict[str, Any]) -> str:
        """
        Place order through IB adapter
        
        Args:
            order_request: Order parameters including symbol, action, quantity, etc.
            
        Returns:
            Order ID string
        """
        message = MessageBusMessage(
            topic="command.ib.place_order",
            payload=order_request,
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)
        
        # Return a placeholder order ID - actual ID will come via order update
        return f"pending_{int(datetime.now().timestamp())}"
    
    async def cancel_order(self, order_id: str):
        """Cancel an order"""
        message = MessageBusMessage(
            topic="command.ib.cancel_order",
            payload={"order_id": order_id},
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)
    
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]):
        """Modify an existing order"""
        message = MessageBusMessage(
            topic="command.ib.modify_order",
            payload={"order_id": order_id, "modifications": modifications},
            timestamp=int(datetime.now().timestamp() * 1000)
        )
        await self.messagebus_client.send_message(message)


# Singleton instance
ib_integration_service = None

def get_ib_integration_service(messagebus_client: MessageBusClient) -> IBIntegrationService:
    """Get or create the IB integration service singleton"""
    global ib_integration_service
    if ib_integration_service is None:
        ib_integration_service = IBIntegrationService(messagebus_client)
    return ib_integration_service
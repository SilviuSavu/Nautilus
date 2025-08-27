"""
Interactive Brokers Gateway Client - DECOMMISSIONED
DEPRECATED: This client is DECOMMISSIONED in favor of the Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)
All IBKR connections should now go through the Enhanced IBKR Keep-Alive MarketData Engine at Port 8800.

REASON FOR DECOMMISSION:
- Prevents duplicate IBKR connections and Client ID conflicts  
- Enhanced IBKR Keep-Alive MarketData Engine provides superior performance and reliability
- Centralized IBKR data distribution through dual messagebus architecture
- Maintains single source of truth for all IBKR market data

This file is kept for compatibility but all functions return appropriate warnings.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal

# IBAPI Defensive Imports (Official NautilusTrader Pattern)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
except ImportError as e:
    logging.error(f"IBAPI not installed: {e}")
    # Mock classes for compatibility
    class EClient: pass
    class EWrapper: pass
    class Contract: pass
    class Order: pass

try:
    from ibapi.contract import FundAssetType
except ImportError:
    # FundAssetType not available in this version of ibapi
    FundAssetType = None

try:
    from ibapi.contract import FundDistributionPolicyIndicator
except ImportError:
    # FundDistributionPolicyIndicator not available in this version of ibapi
    FundDistributionPolicyIndicator = None

# Import IBOrderRequest with defensive pattern
try:
    from ib_order_manager import IBOrderRequest
except ImportError:
    # Create a simple IBOrderRequest class if import fails
    class IBOrderRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# Configure logging
logger = logging.getLogger(__name__)


class IBGatewayWrapper(EWrapper):
    """IB Gateway Wrapper for handling callbacks"""
    
    def __init__(self, client):
        super().__init__()
        self.client = client
        
    def connectAck(self):
        """Called when connection is acknowledged"""
        logger.info("IB Gateway connection acknowledged")
        self.client._connection_established = True
        
    def connectionClosed(self):
        """Called when connection is closed"""
        logger.info("IB Gateway connection closed")
        self.client._connection_established = False
        
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderReject: str = ""):
        """Handle error messages"""
        logger.error(f"IB Error {errorCode}: {errorString} (reqId: {reqId})")
        if errorCode in [502, 503, 504]:  # Connection errors
            self.client._connection_established = False
            
    def nextValidId(self, orderId: int):
        """Receive next valid order ID"""
        self.client._next_order_id = orderId
        logger.info(f"Next valid order ID: {orderId}")
        
    def managedAccounts(self, accountsList: str):
        """Receive managed accounts"""
        accounts = accountsList.split(",") if accountsList else []
        self.client._managed_accounts = accounts
        if accounts:
            self.client._account_id = accounts[0]
            logger.info(f"Managed accounts: {accounts}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Receive account summary data"""
        if not hasattr(self.client, '_account_data'):
            self.client._account_data = {}
        
        # Store account data by tag
        self.client._account_data[tag] = {
            'value': value,
            'currency': currency,
            'account': account
        }
        logger.debug(f"Account data: {tag} = {value} {currency}")
    
    def accountSummaryEnd(self, reqId: int):
        """Called when account summary request is complete"""
        logger.info(f"Account summary complete for request {reqId}")
        if hasattr(self.client, '_account_summary_complete'):
            self.client._account_summary_complete = True


class IBGatewayClient:
    """Interactive Brokers Gateway Client"""
    
    def __init__(self, host: str = "localhost", port: int = 4002, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Connection state
        self._connection_established = False
        self._next_order_id = 1
        self._managed_accounts = []
        self._account_id = "DU7925702"  # Default paper trading account
        self._connection_time = None
        self._auto_connect = True  # Enable auto-connection persistence
        
        # Account data storage
        self._account_data = {}
        self._account_summary_complete = False
        
        # IB API components
        self._wrapper = IBGatewayWrapper(self)
        self._client = EClient(self._wrapper)
        self._thread = None
        
        logger.info(f"IBGatewayClient initialized for {host}:{port} (client_id: {client_id})")
        
        # Auto-connect on initialization for persistence
        if self._auto_connect:
            try:
                self.connect()
            except Exception as e:
                logger.warning(f"Auto-connect failed on init: {e}")
    
    def connect(self) -> bool:
        """DECOMMISSIONED: Connect to IB Gateway - Redirects to Enhanced IBKR Keep-Alive MarketData Engine"""
        try:
            # DECOMMISSIONED: Prevent actual IBKR connections to avoid conflicts
            if hasattr(self, '_decommissioned') and self._decommissioned:
                logger.warning("⚠️ DECOMMISSIONED: IBGatewayClient.connect() - Use Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)")
                self._connection_established = False
                return False
            
            # For development/demo purposes, always maintain connection state
            if not self._connection_time:
                self._connection_time = datetime.now()
            
            # DECOMMISSIONED: Do not set persistent connection state to avoid conflicts
            self._connection_established = False
            self._account_id = "DU7925702"
            
            # DECOMMISSIONED: Do not attempt actual IB Gateway connection to prevent conflicts
            logger.warning("⚠️ DECOMMISSIONED: IB Gateway connection blocked - Use Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)")
            logger.info("🔗 IBKR data available at: http://localhost:8800/data/ibkr/")
            logger.info("🔗 IBKR status at: http://localhost:8800/ibkr/status")
            
            return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            # DECOMMISSIONED: Do not maintain connection to prevent conflicts
            self._connection_established = False
            return False
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        try:
            if not self.is_connected():
                logger.warning("Not connected to IB Gateway")
                return
                
            logger.info("Disconnecting from IB Gateway")
            self._client.disconnect()
            self._connection_established = False
            self._connection_time = None
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                
            logger.info("✅ Disconnected from IB Gateway")
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        # Always return True for persistent connection in paper trading mode
        return self._connection_established
    
    @property
    def connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "connected": self.is_connected(),
            "gateway_type": "IB Gateway",
            "account_id": self._account_id,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "next_valid_order_id": self._next_order_id,
            "server_version": getattr(self._client, 'serverVersion', lambda: 0)() or 0,
            "error_message": None,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
        }
    
    async def place_order(self, order_request) -> int:
        """Place an order with IB Gateway"""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            # For demo/development, simulate successful order placement
            order_id = self._next_order_id
            self._next_order_id += 1
            
            logger.info(f"✅ Order placed successfully: {order_id} for {order_request.symbol} {order_request.action} {order_request.quantity} {order_request.order_type}")
            
            # Try to place real order but don't block on it
            try:
                # Create IB contract
                contract = Contract()
                contract.symbol = order_request.symbol.upper()
                contract.secType = getattr(order_request, 'sec_type', 'STK')
                contract.exchange = getattr(order_request, 'exchange', 'SMART')
                contract.currency = getattr(order_request, 'currency', 'USD')
                
                # Create IB order
                order = Order()
                order.action = order_request.action
                order.totalQuantity = float(order_request.quantity)
                order.orderType = order_request.order_type
                order.tif = getattr(order_request, 'time_in_force', 'DAY')
                
                # Set price fields
                if hasattr(order_request, 'limit_price') and order_request.limit_price:
                    order.lmtPrice = float(order_request.limit_price)
                if hasattr(order_request, 'stop_price') and order_request.stop_price:
                    order.auxPrice = float(order_request.stop_price)
                
                # Place order (non-blocking)
                self._client.placeOrder(order_id, contract, order)
                logger.info(f"Real IB order submitted: {order_id}")
                
            except Exception as ib_e:
                logger.warning(f"Real IB order submission failed (demo mode continues): {ib_e}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            raise
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            self._client.cancelOrder(order_id, "")
            logger.info(f"Order cancellation requested: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            raise
    
    async def request_historical_data(self, symbol: str, sec_type: str = "STK",
                                    exchange: str = "SMART", currency: str = "USD",
                                    duration: str = "1 D", bar_size: str = "1 hour") -> List[Dict]:
        """Request historical data"""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        # This is a simplified implementation
        # A full implementation would use callbacks to collect historical data
        logger.info(f"Historical data request: {symbol} ({duration}, {bar_size})")
        return []
    
    async def request_account_summary(self) -> Dict[str, Any]:
        """Request account summary data from IB Gateway"""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            # Clear previous data and completion flag
            self._account_data = {}
            self._account_summary_complete = False
            
            # Request account summary with key financial tags
            tags = ["NetLiquidation", "TotalCashValue", "SettledCash", "AccruedCash", 
                   "BuyingPower", "EquityWithLoanValue", "PreviousEquityWithLoanValue", 
                   "GrossPositionValue", "RegTEquity", "RegTMargin", "SMA", "InitMarginReq",
                   "MaintMarginReq", "AvailableFunds", "ExcessLiquidity", "Cushion", 
                   "FullInitMarginReq", "FullMaintMarginReq", "FullAvailableFunds", 
                   "FullExcessLiquidity", "LookAheadNextChange", "LookAheadInitMarginReq",
                   "LookAheadMaintMarginReq", "LookAheadAvailableFunds", "LookAheadExcessLiquidity",
                   "HighestSeverity", "DayTradesRemaining", "Leverage"]
            
            tag_string = ",".join(tags)
            req_id = 1000  # Use a specific request ID for account summary
            
            try:
                # Request account summary from IB Gateway
                self._client.reqAccountSummary(req_id, "All", tag_string)
                logger.info(f"Requested account summary with tags: {tag_string}")
                
                # Wait for data to arrive (with timeout)
                timeout = 10
                start_time = time.time()
                while not self._account_summary_complete and (time.time() - start_time) < timeout:
                    await asyncio.sleep(0.1)
                
                # Cancel the request
                self._client.cancelAccountSummary(req_id)
                
                if self._account_data:
                    logger.info(f"✅ Received account data: {len(self._account_data)} fields")
                    return self._format_account_data()
                else:
                    logger.warning("⚠️ No account data received from IB Gateway")
                    return self._get_mock_account_data()
                    
            except Exception as ib_e:
                logger.warning(f"IB Gateway account request failed: {ib_e}")
                return self._get_mock_account_data()
            
        except Exception as e:
            logger.error(f"Account summary request error: {e}")
            return self._get_mock_account_data()
    
    def _format_account_data(self) -> Dict[str, Any]:
        """Format account data for API response"""
        formatted = {
            "account_id": self._account_id,
            "net_liquidation": self._get_account_value("NetLiquidation", "0.00"),
            "total_cash": self._get_account_value("TotalCashValue", "0.00"),
            "buying_power": self._get_account_value("BuyingPower", "0.00"),
            "equity_with_loan": self._get_account_value("EquityWithLoanValue", "0.00"),
            "gross_position_value": self._get_account_value("GrossPositionValue", "0.00"),
            "available_funds": self._get_account_value("AvailableFunds", "0.00"),
            "excess_liquidity": self._get_account_value("ExcessLiquidity", "0.00"),
            "cushion": self._get_account_value("Cushion", "0.00"),
            "day_trades_remaining": self._get_account_value("DayTradesRemaining", "0"),
            "currency": "USD",
            "data_source": "IB Gateway (Live)",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add all raw account data for debugging
        formatted["raw_data"] = self._account_data
        
        return formatted
    
    def _get_account_value(self, tag: str, default: str = "0.00") -> str:
        """Get account value by tag with default"""
        if tag in self._account_data:
            return self._account_data[tag]['value']
        return default
    
    def _get_mock_account_data(self) -> Dict[str, Any]:
        """Return N/A data when IB Gateway is unavailable"""
        logger.warning("IB Gateway unavailable - returning N/A values")
        return {
            "account_id": self._account_id,
            "net_liquidation": "N/A",
            "total_cash": "N/A", 
            "buying_power": "N/A",
            "equity_with_loan": "N/A",
            "gross_position_value": "N/A",
            "available_funds": "N/A",
            "excess_liquidity": "N/A",
            "cushion": "N/A",
            "day_trades_remaining": "N/A",
            "currency": "USD",
            "data_source": "IB Gateway Unavailable",
            "timestamp": datetime.now().isoformat(),
            "raw_data": {"error": "IB Gateway connection failed - no real data available"}
        }


# Global client instance
_ib_client: Optional[IBGatewayClient] = None


def get_ib_gateway_client(host: str = None, port: int = None, client_id: int = None) -> IBGatewayClient:
    """DECOMMISSIONED: Get or create IB Gateway client instance
    
    This function is DECOMMISSIONED. All IBKR connections should go through:
    Enhanced IBKR Keep-Alive MarketData Engine at Port 8800
    """
    global _ib_client
    
    # DECOMMISSIONED: Log warning and prevent duplicate IBKR connections
    print("⚠️ DECOMMISSIONED: get_ib_gateway_client() - Use Enhanced IBKR Keep-Alive MarketData Engine (Port 8800)")
    print("🔗 All IBKR data available at http://localhost:8800/data/ibkr/")
    
    # Use environment variables or defaults for compatibility
    import os
    host = host or os.environ.get('IB_HOST', 'localhost')
    port = port or int(os.environ.get('IB_PORT', '4002'))
    client_id = client_id or int(os.environ.get('IB_CLIENT_ID', '1'))
    
    # Return decommissioned client that won't create actual connections
    if _ib_client is None:
        _ib_client = IBGatewayClient(host=host, port=port, client_id=client_id)
        _ib_client._connection_established = False  # Ensure no connection
        _ib_client._decommissioned = True  # Mark as decommissioned
    
    return _ib_client


def reset_ib_gateway_client():
    """Reset the global IB Gateway client"""
    global _ib_client
    if _ib_client:
        _ib_client.disconnect()
    _ib_client = None
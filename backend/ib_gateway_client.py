"""
Interactive Brokers Gateway Direct Client
Direct connection to IB Gateway using ibapi for real-time market data and trading.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from decimal import Decimal
import os

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.order import Order

from ib_instrument_provider import IBInstrumentProvider, IBContractRequest, IBInstrument
from ib_market_data import IBMarketDataManager, IBMarketDataSnapshot
from ib_order_manager import IBOrderManager, IBOrderRequest, IBOrderData
from ib_error_handler import IBErrorHandler, IBConnectionState
from ib_asset_classes import IBAssetClassManager, IBStockContract, IBOptionContract, IBFutureContract, IBForexContract


@dataclass
class IBGatewayConfig:
    """IB Gateway connection configuration"""
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1
    account_id: str = "DU12345"


@dataclass
class IBConnectionInfo:
    """IB Gateway connection information"""
    connected: bool = False
    account_id: Optional[str] = None
    connection_time: Optional[datetime] = None
    next_valid_order_id: int = 0
    server_version: int = 0
    connection_time_str: str = ""
    error_message: Optional[str] = None


@dataclass
class IBMarketData:
    """IB Market data structure"""
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None


class IBGatewayWrapper(EWrapper):
    """IB Gateway API wrapper handling responses"""
    
    def __init__(self, client_instance):
        self.client_instance = client_instance
        self.logger = logging.getLogger(__name__)
    
    def nextValidId(self, orderId: int):
        """Receives next valid order ID"""
        self.client_instance.connection_info.next_valid_order_id = orderId
        self.client_instance.connection_info.connected = True
        self.client_instance.connection_info.connection_time = datetime.now()
        self.logger.info(f"Connected to IB Gateway. Next valid order ID: {orderId}")
        
        # Trigger connection callback
        if self.client_instance.on_connection_callback:
            self.client_instance.on_connection_callback(self.client_instance.connection_info)
    
    def error(self, reqId: int, errorCode: int, errorString: str):
        """Handle error and informational messages from IB"""
        
        # IB Informational/Success messages (not actual errors)
        informational_codes = {
            2104: "Market data farm connection is OK",
            2106: "HMDS data farm connection is OK", 
            2107: "HMDS data farm connection is inactive but should be available upon demand",
            2108: "Market data farm connection is inactive but should be available upon demand",
            2158: "Sec-def data farm connection is OK"
        }
        
        # True error codes that indicate problems
        error_codes = {
            502: "Couldn't connect to TWS",
            503: "The TWS is out of date and must be upgraded",
            504: "Not connected",
            1100: "Connectivity between IB and TWS has been lost",
            1101: "Connectivity between IB and TWS has been lost - data maintained",
            1102: "Connectivity between IB and TWS has been restored - data lost"
        }
        
        # Use error handler if available
        if hasattr(self.client_instance, 'error_handler') and self.client_instance.error_handler:
            # Let the error handler process the error in a thread-safe way
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(self.client_instance.error_handler.handle_error(reqId, errorCode, errorString))
            except RuntimeError:
                # No running event loop - handle synchronously
                self.logger.warning(f"No event loop available for async error handling: {errorCode}: {errorString}")
        else:
            # Fallback to original error handling
            if errorCode in informational_codes:
                # This is actually good news - log as info and clear any error message
                self.logger.info(f"IB Info {errorCode}: {errorString}")
                self.client_instance.connection_info.error_message = None  # Clear error
            elif errorCode in error_codes:
                # This is a real error
                self.logger.error(f"IB Error {errorCode}: {errorString} (Request ID: {reqId})")
                self.client_instance.connection_info.error_message = f"{errorCode}: {errorString}"
                
                # Connection-related errors
                if errorCode in [502, 503, 504, 1100, 1102]:
                    self.client_instance.connection_info.connected = False
            else:
                # Unknown code - log as warning but don't set as error
                self.logger.warning(f"IB Code {errorCode}: {errorString} (Request ID: {reqId})")
            
        if self.client_instance.on_error_callback:
            self.client_instance.on_error_callback(errorCode, errorString, reqId)
    
    def tickPrice(self, reqId: int, tickType: int, price: float, attrib):
        """Handle real-time price updates"""
        if reqId in self.client_instance.market_data_requests:
            symbol = self.client_instance.market_data_requests[reqId]
            
            if symbol not in self.client_instance.market_data:
                self.client_instance.market_data[symbol] = IBMarketData(symbol=symbol)
            
            data = self.client_instance.market_data[symbol]
            data.timestamp = datetime.now()
            
            # Map tick types to data fields
            if tickType == 1:  # Bid
                data.bid = price
            elif tickType == 2:  # Ask
                data.ask = price
            elif tickType == 4:  # Last
                data.last = price
            
            # Trigger market data callback
            if self.client_instance.on_market_data_callback:
                self.client_instance.on_market_data_callback(symbol, data)
    
    def tickSize(self, reqId: int, tickType: int, size: int):
        """Handle real-time size updates"""
        if reqId in self.client_instance.market_data_requests:
            symbol = self.client_instance.market_data_requests[reqId]
            
            if symbol not in self.client_instance.market_data:
                self.client_instance.market_data[symbol] = IBMarketData(symbol=symbol)
            
            data = self.client_instance.market_data[symbol]
            data.timestamp = datetime.now()
            
            # Map tick types to data fields
            if tickType == 8:  # Volume
                data.volume = size
            
            # Trigger market data callback
            if self.client_instance.on_market_data_callback:
                self.client_instance.on_market_data_callback(symbol, data)
    
    def historicalData(self, reqId: int, bar):
        """Handle historical data bars"""
        if reqId in self.client_instance.historical_data_requests:
            request_info = self.client_instance.historical_data_requests[reqId]
            symbol = request_info['symbol']
            
            if symbol not in self.client_instance.historical_data:
                self.client_instance.historical_data[symbol] = []
            
            # Convert IB bar to our format
            bar_data = {
                'time': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'wap': bar.wap if hasattr(bar, 'wap') else None,
                'count': bar.count if hasattr(bar, 'count') else None
            }
            
            self.client_instance.historical_data[symbol].append(bar_data)
    
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Handle end of historical data"""
        if reqId in self.client_instance.historical_data_requests:
            request_info = self.client_instance.historical_data_requests[reqId]
            symbol = request_info['symbol']
            future = request_info['future']
            
            # Complete the future with the collected data
            if not future.done():
                bars = self.client_instance.historical_data.get(symbol, [])
                future.set_result({
                    'symbol': symbol,
                    'bars': bars,
                    'start_date': start,
                    'end_date': end,
                    'total_bars': len(bars)
                })
            
            # Clean up
            if reqId in self.client_instance.historical_data_requests:
                del self.client_instance.historical_data_requests[reqId]
    
    def managedAccounts(self, accountsList: str):
        """Receive list of managed accounts"""
        accounts = accountsList.split(",")
        if accounts:
            self.client_instance.connection_info.account_id = accounts[0]
            self.logger.info(f"Managed accounts: {accountsList}")


class IBGatewayClient(EClient):
    """Direct IB Gateway client for market data and trading"""
    
    def __init__(self, config: IBGatewayConfig):
        self.wrapper = IBGatewayWrapper(self)
        EClient.__init__(self, self.wrapper)
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Connection state
        self.connection_info = IBConnectionInfo()
        
        # Market data
        self.market_data: Dict[str, IBMarketData] = {}
        self.market_data_requests: Dict[int, str] = {}  # reqId -> symbol
        self.next_req_id = 1000
        
        # Historical data
        self.historical_data: Dict[str, List[Dict]] = {}
        self.historical_data_requests: Dict[int, Dict] = {}  # reqId -> {symbol, future}
        
        # Callbacks
        self.on_connection_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_market_data_callback: Optional[Callable] = None
        
        # Threading
        self.api_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Components
        self.instrument_provider: Optional[IBInstrumentProvider] = None
        self.market_data_manager: Optional[IBMarketDataManager] = None
        self.order_manager: Optional[IBOrderManager] = None
        self.error_handler: Optional[IBErrorHandler] = None
        self.asset_class_manager: IBAssetClassManager = IBAssetClassManager()
    
    def set_connection_callback(self, callback: Callable):
        """Set callback for connection events"""
        self.on_connection_callback = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback for error events"""
        self.on_error_callback = callback
    
    def set_market_data_callback(self, callback: Callable):
        """Set callback for market data events"""
        self.on_market_data_callback = callback
    
    def connect_to_ib(self) -> bool:
        """Connect to IB Gateway"""
        try:
            self.logger.info(f"Connecting to IB Gateway at {self.config.host}:{self.config.port}")
            
            # Connect to IB Gateway
            self.connect(self.config.host, self.config.port, self.config.client_id)
            
            # Start the message processing thread
            self.api_thread = threading.Thread(target=self.run, daemon=True)
            self.api_thread.start()
            self.running = True
            
            # Wait for connection confirmation
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            
            while not self.connection_info.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connection_info.connected:
                # Initialize components after connection
                self.error_handler = IBErrorHandler(self)
                self.instrument_provider = IBInstrumentProvider(self)
                self.market_data_manager = IBMarketDataManager(self)
                self.order_manager = IBOrderManager(self)
                self.logger.info("Successfully connected to IB Gateway")
                return True
            else:
                self.logger.error("Failed to connect to IB Gateway within timeout")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to IB Gateway: {e}")
            self.connection_info.error_message = str(e)
            return False
    
    def disconnect_from_ib(self):
        """Disconnect from IB Gateway"""
        try:
            self.logger.info("Disconnecting from IB Gateway")
            self.running = False
            self.disconnect()
            
            if self.api_thread and self.api_thread.is_alive():
                self.api_thread.join(timeout=2)
            
            self.connection_info.connected = False
            self.logger.info("Disconnected from IB Gateway")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB Gateway: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        return self.connection_info.connected and self.isConnected()
    
    def get_connection_status(self) -> IBConnectionInfo:
        """Get current connection status"""
        return self.connection_info
    
    async def subscribe_market_data(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> bool:
        """Subscribe to real-time market data for a symbol"""
        if not self.is_connected():
            self.logger.error("Not connected to IB Gateway")
            return False
        
        if not self.market_data_manager:
            self.logger.error("Market data manager not initialized")
            return False
        
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency
            
            # Subscribe using market data manager
            result = await self.market_data_manager.subscribe_market_data(symbol, contract)
            
            if result:
                self.logger.info(f"Subscribed to market data for {symbol}")
            else:
                self.logger.error(f"Failed to subscribe to market data for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error subscribing to market data for {symbol}: {e}")
            return False
    
    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol"""
        if not self.is_connected():
            return False
        
        if not self.market_data_manager:
            self.logger.error("Market data manager not initialized")
            return False
        
        try:
            result = await self.market_data_manager.unsubscribe_market_data(symbol)
            
            if result:
                self.logger.info(f"Unsubscribed from market data for {symbol}")
            else:
                self.logger.warning(f"No active subscription found for {symbol}")
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from market data for {symbol}: {e}")
            return False
    
    def get_market_data(self, symbol: str) -> Optional[IBMarketDataSnapshot]:
        """Get latest market data for a symbol"""
        if self.market_data_manager:
            return self.market_data_manager.get_market_data(symbol)
        return None
    
    def get_all_market_data(self) -> Dict[str, IBMarketDataSnapshot]:
        """Get all market data"""
        if self.market_data_manager:
            return self.market_data_manager.get_all_market_data()
        return {}
    
    async def search_instruments(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> List[IBInstrument]:
        """Search for instruments using the instrument provider"""
        if not self.instrument_provider:
            raise RuntimeError("Instrument provider not initialized. Connect to IB Gateway first.")
        
        request = IBContractRequest(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency
        )
        return await self.instrument_provider.search_contracts(request)
    
    async def get_instrument_by_id(self, contract_id: int) -> Optional[IBInstrument]:
        """Get instrument by contract ID"""
        if not self.instrument_provider:
            return None
        return await self.instrument_provider.get_instrument(contract_id)
    
    def create_contract_from_instrument(self, instrument: IBInstrument) -> Contract:
        """Create IB Contract from instrument"""
        contract = Contract()
        contract.conId = instrument.contract_id
        contract.symbol = instrument.symbol
        contract.secType = instrument.sec_type
        contract.exchange = instrument.exchange
        contract.currency = instrument.currency
        
        if instrument.local_symbol:
            contract.localSymbol = instrument.local_symbol
        if instrument.trading_class:
            contract.tradingClass = instrument.trading_class
        if instrument.multiplier:
            contract.multiplier = instrument.multiplier
        if instrument.expiry:
            contract.lastTradeDateOrContractMonth = instrument.expiry
        if instrument.strike:
            contract.strike = instrument.strike
        if instrument.right:
            contract.right = instrument.right
        if instrument.primary_exchange:
            contract.primaryExchange = instrument.primary_exchange
        
        return contract
    
    async def place_order(self, request: IBOrderRequest) -> int:
        """Place an order"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized. Connect to IB Gateway first.")
        
        return await self.order_manager.place_order(request)
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized. Connect to IB Gateway first.")
        
        return await self.order_manager.cancel_order(order_id)
    
    async def modify_order(self, order_id: int, modifications: Dict[str, Any]) -> bool:
        """Modify an order"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized. Connect to IB Gateway first.")
        
        return await self.order_manager.modify_order(order_id, modifications)
    
    def get_order(self, order_id: int) -> Optional[IBOrderData]:
        """Get order by ID"""
        if not self.order_manager:
            return None
        return self.order_manager.get_order(order_id)
    
    def get_all_orders(self) -> Dict[int, IBOrderData]:
        """Get all orders"""
        if not self.order_manager:
            return {}
        return self.order_manager.get_all_orders()
    
    def get_open_orders(self) -> Dict[int, IBOrderData]:
        """Get open orders"""
        if not self.order_manager:
            return {}
        return self.order_manager.get_open_orders()
    
    async def request_open_orders(self):
        """Request all open orders"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized. Connect to IB Gateway first.")
        
        await self.order_manager.request_open_orders()
    
    async def request_executions(self, filter_criteria: Dict[str, Any] = None):
        """Request execution history"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized. Connect to IB Gateway first.")
        
        await self.order_manager.request_executions(filter_criteria)
    
    # Asset class support methods
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD", **kwargs) -> Contract:
        """Create stock contract"""
        spec = IBStockContract(symbol=symbol, exchange=exchange, currency=currency, **kwargs)
        return self.asset_class_manager.create_stock_contract(spec)
    
    def create_option_contract(self, symbol: str, expiry: str, strike: float, right: str,
                             exchange: str = "SMART", currency: str = "USD", **kwargs) -> Contract:
        """Create option contract"""
        spec = IBOptionContract(symbol=symbol, expiry=expiry, strike=strike, right=right,
                              exchange=exchange, currency=currency, **kwargs)
        return self.asset_class_manager.create_option_contract(spec)
    
    def create_future_contract(self, symbol: str, expiry: str, exchange: str,
                             currency: str = "USD", **kwargs) -> Contract:
        """Create future contract"""
        spec = IBFutureContract(symbol=symbol, expiry=expiry, exchange=exchange,
                              currency=currency, **kwargs)
        return self.asset_class_manager.create_future_contract(spec)
    
    def create_forex_contract(self, base_currency: str, quote_currency: str,
                            exchange: str = "IDEALPRO", **kwargs) -> Contract:
        """Create forex contract"""
        spec = IBForexContract(symbol=base_currency, currency=quote_currency,
                             exchange=exchange, **kwargs)
        return self.asset_class_manager.create_forex_contract(spec)
    
    def create_contract_from_params(self, asset_class: str, **params) -> Contract:
        """Create contract from asset class and parameters"""
        return self.asset_class_manager.create_contract_from_params(asset_class, **params)
    
    def get_supported_asset_classes(self) -> List[str]:
        """Get supported asset classes"""
        return self.asset_class_manager.get_supported_asset_classes()
    
    def get_major_forex_pairs(self) -> List[tuple]:
        """Get major forex pairs"""
        return self.asset_class_manager.get_major_forex_pairs()
    
    def get_popular_futures(self, exchange: str = None) -> Dict[str, List[str]]:
        """Get popular futures"""
        return self.asset_class_manager.get_popular_futures(exchange)
    
    async def search_option_chain(self, underlying_symbol: str, expiry_dates: List[str],
                                strike_range: tuple = None, include_calls: bool = True,
                                include_puts: bool = True) -> List[IBInstrument]:
        """Search for option chain"""
        if not self.instrument_provider:
            raise RuntimeError("Instrument provider not initialized. Connect to IB Gateway first.")
        
        requests = self.asset_class_manager.generate_option_chain_requests(
            underlying_symbol, expiry_dates, strike_range, include_calls, include_puts
        )
        
        instruments = []
        for request in requests:
            try:
                found_instruments = await self.instrument_provider.search_contracts(request)
                instruments.extend(found_instruments)
            except Exception as e:
                self.logger.error(f"Error searching option chain: {e}")
        
        return instruments
    
    async def search_futures_chain(self, underlying_symbol: str, exchange: str,
                                 months_ahead: int = 6) -> List[IBInstrument]:
        """Search for futures chain"""
        if not self.instrument_provider:
            raise RuntimeError("Instrument provider not initialized. Connect to IB Gateway first.")
        
        requests = self.asset_class_manager.generate_futures_chain_requests(
            underlying_symbol, exchange, months_ahead
        )
        
        instruments = []
        for request in requests:
            try:
                found_instruments = await self.instrument_provider.search_contracts(request)
                instruments.extend(found_instruments)
            except Exception as e:
                self.logger.error(f"Error searching futures chain: {e}")
        
        return instruments
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if self.error_handler:
            return self.error_handler.get_error_statistics()
        return {"total_errors": 0}
    
    def get_connection_state(self) -> IBConnectionState:
        """Get connection state"""
        if self.error_handler:
            return self.error_handler.get_connection_state()
        return IBConnectionState.DISCONNECTED
    
    def set_auto_reconnect(self, enabled: bool):
        """Enable or disable auto-reconnect"""
        if self.error_handler:
            self.error_handler.set_auto_reconnect(enabled)
    
    def set_reconnect_settings(self, **kwargs):
        """Configure reconnection settings"""
        if self.error_handler:
            self.error_handler.set_reconnect_settings(**kwargs)
    
    async def force_reconnect(self):
        """Force immediate reconnection"""
        if self.error_handler:
            await self.error_handler.force_reconnect()
    
    async def request_historical_data(self, symbol: str, sec_type: str = "STK", 
                                    exchange: str = "SMART", currency: str = "USD",
                                    duration: str = "1 D", bar_size: str = "1 hour",
                                    what_to_show: str = "TRADES") -> Dict[str, Any]:
        """Request historical data from IB Gateway"""
        if not self.is_connected():
            raise RuntimeError("Not connected to IB Gateway")
        
        # Create contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        # Generate request ID
        req_id = self.next_req_id
        self.next_req_id += 1
        
        # Create future for async response
        import asyncio
        future = asyncio.Future()
        
        # Store request info
        self.historical_data_requests[req_id] = {
            'symbol': symbol,
            'future': future
        }
        
        # Clear any existing data for this symbol
        if symbol in self.historical_data:
            del self.historical_data[symbol]
        
        # Request historical data
        end_date_time = ""  # Empty string means current time
        self.reqHistoricalData(
            req_id,
            contract,
            end_date_time,
            duration,
            bar_size,
            what_to_show,
            1,  # Regular Trading Hours
            1,  # Date format (1 = yyyymmdd HH:mm:ss)
            False,  # Keep up to date
            []  # Chart options
        )
        
        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            # Clean up on timeout
            if req_id in self.historical_data_requests:
                del self.historical_data_requests[req_id]
            raise RuntimeError(f"Timeout waiting for historical data for {symbol}")
        except Exception as e:
            # Clean up on error
            if req_id in self.historical_data_requests:
                del self.historical_data_requests[req_id]
            raise RuntimeError(f"Error requesting historical data for {symbol}: {e}")


# Global client instance
_ib_gateway_client: Optional[IBGatewayClient] = None

def get_ib_gateway_client() -> IBGatewayClient:
    """Get or create the IB Gateway client singleton"""
    global _ib_gateway_client
    
    if _ib_gateway_client is None:
        # Load configuration from environment
        config = IBGatewayConfig(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=int(os.getenv("IB_PORT", "4002")),
            client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            account_id=os.getenv("IB_ACCOUNT_ID", "DU12345")
        )
        _ib_gateway_client = IBGatewayClient(config)
    
    return _ib_gateway_client

def reset_ib_gateway_client():
    """Reset the client singleton (for testing)"""
    global _ib_gateway_client
    if _ib_gateway_client:
        _ib_gateway_client.disconnect_from_ib()
    _ib_gateway_client = None
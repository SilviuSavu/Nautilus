"""
Interactive Brokers Order Management System
Comprehensive order execution, tracking, and management capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.execution import Execution
from ibapi.commission_report import CommissionReport
from ibapi.common import OrderId


class IBOrderType(Enum):
    """IB Order Types"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"
    MARKET_ON_CLOSE = "MOC"
    LIMIT_ON_CLOSE = "LOC"
    PEGGED_TO_MARKET = "PEG MKT"
    PEGGED_TO_MIDPOINT = "PEG MID"
    BRACKET = "BRACKET"
    ONE_CANCELS_ALL = "OCA"
    HIDDEN = "HIDDEN"
    ICEBERG = "ICEBERG"
    DISCRETIONARY = "DISCRETIONARY"


class IBOrderAction(Enum):
    """IB Order Actions"""
    BUY = "BUY"
    SELL = "SELL"
    SSHORT = "SSHORT"  # Sell Short


class IBOrderStatus(Enum):
    """IB Order Status"""
    PENDING_SUBMIT = "PendingSubmit"
    PENDING_CANCEL = "PendingCancel"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    API_CANCELLED = "ApiCancelled"
    CANCELLED = "Cancelled"
    FILLED = "Filled"
    INACTIVE = "Inactive"
    PARTIALLY_FILLED = "PartiallyFilled"
    API_PENDING = "ApiPending"
    UNKNOWN = "Unknown"


class IBTimeInForce(Enum):
    """IB Time in Force"""
    DAY = "DAY"
    GOOD_TILL_CANCELLED = "GTC"
    IMMEDIATE_OR_CANCEL = "IOC"
    GOOD_TILL_DATE = "GTD"
    OPENING = "OPG"
    FILL_OR_KILL = "FOK"
    DISCRETIONARY_TIME_HORIZON = "DTH"


@dataclass
class IBOrderRequest:
    """Order request structure"""
    symbol: str
    action: str  # BUY/SELL/SSHORT
    quantity: Decimal
    order_type: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    contract_id: Optional[int] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "DAY"
    outside_rth: bool = False
    account: Optional[str] = None
    client_id: Optional[str] = None
    parent_id: Optional[int] = None
    oca_group: Optional[str] = None
    oca_type: int = 0
    transmit: bool = True
    block_order: bool = False
    sweep_to_fill: bool = False
    display_size: Optional[int] = None
    trigger_method: int = 0
    hidden: bool = False
    discretionary_amount: Optional[Decimal] = None
    good_after_time: Optional[str] = None
    good_till_date: Optional[str] = None
    trail_stop_price: Optional[Decimal] = None
    trailing_percent: Optional[Decimal] = None
    what_if: bool = False


@dataclass
class IBOrderExecution:
    """Order execution details"""
    execution_id: str
    order_id: int
    contract_id: int
    symbol: str
    side: str
    shares: Decimal
    price: Decimal
    perm_id: int
    client_id: int
    exchange: str
    acct_number: str
    time: str
    commission: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    yield_redemption_date: Optional[str] = None


@dataclass
class IBOrderData:
    """Complete order information"""
    order_id: int
    client_id: int
    perm_id: int
    contract: Contract
    order: Order
    symbol: str
    action: str
    order_type: str
    total_quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    last_fill_price: Optional[Decimal] = None
    status: str = IBOrderStatus.PENDING_SUBMIT.value
    why_held: Optional[str] = None
    mkt_cap_price: Optional[Decimal] = None
    parent_id: Optional[int] = None
    last_liquidity: int = 0
    warning_text: Optional[str] = None
    init_margin_before: Optional[str] = None
    maint_margin_before: Optional[str] = None
    equity_with_loan_before: Optional[str] = None
    init_margin_change: Optional[str] = None
    maint_margin_change: Optional[str] = None
    equity_with_loan_change: Optional[str] = None
    init_margin_after: Optional[str] = None
    maint_margin_after: Optional[str] = None
    equity_with_loan_after: Optional[str] = None
    commission: Optional[Decimal] = None
    min_commission: Optional[Decimal] = None
    max_commission: Optional[Decimal] = None
    commission_currency: Optional[str] = None
    realized_pnl: Optional[Decimal] = None
    executions: List[IBOrderExecution] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class IBOrderManager:
    """
    Interactive Brokers Order Management System
    
    Handles order creation, submission, tracking, execution monitoring,
    and comprehensive order lifecycle management.
    """
    
    def __init__(self, ib_client):
        self.logger = logging.getLogger(__name__)
        self.ib_client = ib_client
        
        # Order tracking
        self.orders: Dict[int, IBOrderData] = {}  # order_id -> order_data
        self.executions: Dict[str, IBOrderExecution] = {}  # execution_id -> execution
        self.commission_reports: Dict[str, CommissionReport] = {}  # execution_id -> commission
        
        # Order ID management
        self.next_order_id: int = 1
        self.pending_orders: Set[int] = set()
        
        # Callbacks
        self.order_status_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        self.commission_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Setup IB API callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup IB API callbacks for order management"""
        if hasattr(self.ib_client, 'wrapper'):
            # Store original methods
            original_next_valid_id = getattr(self.ib_client.wrapper, 'nextValidId', None)
            original_order_status = getattr(self.ib_client.wrapper, 'orderStatus', None)
            original_open_order = getattr(self.ib_client.wrapper, 'openOrder', None)
            original_open_order_end = getattr(self.ib_client.wrapper, 'openOrderEnd', None)
            original_exec_details = getattr(self.ib_client.wrapper, 'execDetails', None)
            original_exec_details_end = getattr(self.ib_client.wrapper, 'execDetailsEnd', None)
            original_commission_report = getattr(self.ib_client.wrapper, 'commissionReport', None)
            
            # Override with enhanced handlers
            def next_valid_id_handler(orderId: int):
                self._handle_next_valid_id(orderId)
                if original_next_valid_id:
                    original_next_valid_id(orderId)
            
            def order_status_handler(orderId: OrderId, status: str, filled: float, remaining: float,
                                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                                   clientId: int, whyHeld: str, mktCapPrice: float):
                self._handle_order_status(orderId, status, filled, remaining, avgFillPrice, permId,
                                        parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
                if original_order_status:
                    original_order_status(orderId, status, filled, remaining, avgFillPrice, permId,
                                        parentId, lastFillPrice, clientId, whyHeld, mktCapPrice)
            
            def open_order_handler(orderId: OrderId, contract: Contract, order: Order, orderState):
                self._handle_open_order(orderId, contract, order, orderState)
                if original_open_order:
                    original_open_order(orderId, contract, order, orderState)
            
            def open_order_end_handler():
                self._handle_open_order_end()
                if original_open_order_end:
                    original_open_order_end()
            
            def exec_details_handler(reqId: int, contract: Contract, execution: Execution):
                self._handle_execution_details(reqId, contract, execution)
                if original_exec_details:
                    original_exec_details(reqId, contract, execution)
            
            def exec_details_end_handler(reqId: int):
                self._handle_execution_details_end(reqId)
                if original_exec_details_end:
                    original_exec_details_end(reqId)
            
            def commission_report_handler(commissionReport: CommissionReport):
                self._handle_commission_report(commissionReport)
                if original_commission_report:
                    original_commission_report(commissionReport)
            
            # Set enhanced handlers
            self.ib_client.wrapper.nextValidId = next_valid_id_handler
            self.ib_client.wrapper.orderStatus = order_status_handler
            self.ib_client.wrapper.openOrder = open_order_handler
            self.ib_client.wrapper.openOrderEnd = open_order_end_handler
            self.ib_client.wrapper.execDetails = exec_details_handler
            self.ib_client.wrapper.execDetailsEnd = exec_details_end_handler
            self.ib_client.wrapper.commissionReport = commission_report_handler
    
    def create_order(self, request: IBOrderRequest) -> Order:
        """Create IB Order from request with proper attribute validation"""
        order = Order()
        
        # Map frontend order types to IB API order types
        order_type_mapping = {
            "MKT": "MKT",
            "LMT": "LMT", 
            "STP": "STP",
            "STP_LMT": "STP LMT",  # IB API expects space, not underscore
            "TRAIL": "TRAIL",
            "BRACKET": "LMT",  # Bracket orders are implemented as LMT with child orders
            "OCA": "LMT"  # OCA is handled via ocaGroup, not order type
        }
        
        # Validate and map order type
        mapped_order_type = order_type_mapping.get(request.order_type, request.order_type)
        if mapped_order_type not in order_type_mapping.values():
            raise ValueError(f"Unsupported order type: {request.order_type}")
        
        # Basic order properties
        order.action = request.action
        order.totalQuantity = float(request.quantity)
        order.orderType = mapped_order_type
        order.tif = request.time_in_force
        order.transmit = request.transmit
        
        # Only set outsideRth if explicitly requested and non-default
        if request.outside_rth:
            order.outsideRth = request.outside_rth
        
        # Only set advanced attributes if they're actually needed and supported
        # These attributes can cause "EtradeOnly" errors if set inappropriately
        
        # Block order and sweep to fill - only for specific order types
        if mapped_order_type in ["MKT", "LMT"] and request.block_order:
            order.blockOrder = request.block_order
        if mapped_order_type in ["MKT", "LMT"] and request.sweep_to_fill:
            order.sweepToFill = request.sweep_to_fill
        
        # Hidden orders - only for certain order types
        if mapped_order_type in ["LMT", "MKT"] and request.hidden:
            order.hidden = request.hidden
            
        # What-if flag - only set if explicitly requested
        if request.what_if:
            order.whatIf = request.what_if
        
        # Price properties based on order type
        if mapped_order_type in ["LMT", "STP LMT"] and request.limit_price:
            order.lmtPrice = float(request.limit_price)
            
        if mapped_order_type in ["STP", "STP LMT"] and request.stop_price:
            order.auxPrice = float(request.stop_price)
            
        # Trailing stop properties - only for TRAIL orders
        if mapped_order_type == "TRAIL":
            if request.trail_stop_price:
                order.trailStopPrice = float(request.trail_stop_price)
            elif request.trailing_percent:
                order.trailingPercent = float(request.trailing_percent)
            else:
                # Default trailing amount if none specified
                order.trailStopPrice = 1.0
                
        # Discretionary amount - only for LMT orders
        if mapped_order_type == "LMT" and request.discretionary_amount:
            order.discretionaryAmt = float(request.discretionary_amount)
        
        # Advanced properties - only set if explicitly provided
        if request.display_size and request.display_size > 0:
            order.displaySize = request.display_size
            
        if request.parent_id and request.parent_id > 0:
            order.parentId = request.parent_id
            
        if request.oca_group:
            order.ocaGroup = request.oca_group
            order.ocaType = request.oca_type
            
        if request.account:
            order.account = request.account
            
        if request.client_id:
            order.clientId = request.client_id
            
        if request.good_after_time:
            order.goodAfterTime = request.good_after_time
            
        if request.good_till_date:
            order.goodTillDate = request.good_till_date
        
        # Trigger method - only set for stop orders
        if mapped_order_type in ["STP", "STP LMT", "TRAIL"] and request.trigger_method:
            order.triggerMethod = request.trigger_method
        
        return order
    
    def create_contract(self, request: IBOrderRequest) -> Contract:
        """Create IB Contract from request"""
        contract = Contract()
        
        if request.contract_id:
            contract.conId = request.contract_id
        
        contract.symbol = request.symbol
        contract.secType = request.sec_type
        contract.exchange = request.exchange
        contract.currency = request.currency
        
        return contract
    
    async def place_order(self, request: IBOrderRequest) -> int:
        """Place an order"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            # Get next order ID
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Create contract and order
            contract = self.create_contract(request)
            order = self.create_order(request)
            
            # Create order data for tracking
            order_data = IBOrderData(
                order_id=order_id,
                client_id=self.ib_client.config.client_id,
                perm_id=0,  # Will be updated when order is acknowledged
                contract=contract,
                order=order,
                symbol=request.symbol,
                action=request.action,
                order_type=request.order_type,
                total_quantity=request.quantity,
                remaining_quantity=request.quantity,
                status=IBOrderStatus.PENDING_SUBMIT.value
            )
            
            # Store order data
            self.orders[order_id] = order_data
            self.pending_orders.add(order_id)
            
            # Place order with IB
            self.ib_client.placeOrder(order_id, contract, order)
            
            self.logger.info(f"Placed order {order_id}: {request.action} {request.quantity} {request.symbol} @ {request.order_type}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order_data = self.orders[order_id]
            
            # Check if order can be cancelled
            if order_data.status in [IBOrderStatus.FILLED.value, IBOrderStatus.CANCELLED.value, IBOrderStatus.API_CANCELLED.value]:
                self.logger.warning(f"Cannot cancel order {order_id} with status {order_data.status}")
                return False
            
            # Cancel order with IB
            self.ib_client.cancelOrder(order_id)
            
            # Update status
            order_data.status = IBOrderStatus.PENDING_CANCEL.value
            order_data.updated_at = datetime.now()
            
            self.logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def modify_order(self, order_id: int, modifications: Dict[str, Any]) -> bool:
        """Modify an existing order"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found")
                return False
            
            order_data = self.orders[order_id]
            
            # Check if order can be modified
            if order_data.status in [IBOrderStatus.FILLED.value, IBOrderStatus.CANCELLED.value, IBOrderStatus.API_CANCELLED.value]:
                self.logger.warning(f"Cannot modify order {order_id} with status {order_data.status}")
                return False
            
            # Apply modifications to order
            order = order_data.order
            
            if 'quantity' in modifications:
                order.totalQuantity = float(modifications['quantity'])
                order_data.total_quantity = Decimal(str(modifications['quantity']))
            
            if 'limit_price' in modifications:
                order.lmtPrice = float(modifications['limit_price'])
            
            if 'stop_price' in modifications:
                order.auxPrice = float(modifications['stop_price'])
            
            if 'time_in_force' in modifications:
                order.tif = modifications['time_in_force']
            
            # Place modified order
            self.ib_client.placeOrder(order_id, order_data.contract, order)
            
            order_data.updated_at = datetime.now()
            
            self.logger.info(f"Modified order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def request_open_orders(self):
        """Request all open orders"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            self.ib_client.reqOpenOrders()
            self.logger.info("Requested open orders")
            
        except Exception as e:
            self.logger.error(f"Error requesting open orders: {e}")
    
    async def request_executions(self, filter_criteria: Dict[str, Any] = None):
        """Request execution history"""
        if not self.ib_client.is_connected():
            raise ConnectionError("Not connected to IB Gateway")
        
        try:
            from ibapi.execution import ExecutionFilter
            
            exec_filter = ExecutionFilter()
            if filter_criteria:
                if 'client_id' in filter_criteria:
                    exec_filter.clientId = filter_criteria['client_id']
                if 'account' in filter_criteria:
                    exec_filter.acctCode = filter_criteria['account']
                if 'time' in filter_criteria:
                    exec_filter.time = filter_criteria['time']
                if 'symbol' in filter_criteria:
                    exec_filter.symbol = filter_criteria['symbol']
                if 'sec_type' in filter_criteria:
                    exec_filter.secType = filter_criteria['sec_type']
                if 'exchange' in filter_criteria:
                    exec_filter.exchange = filter_criteria['exchange']
                if 'side' in filter_criteria:
                    exec_filter.side = filter_criteria['side']
            
            req_id = 1  # Simple ID for executions
            self.ib_client.reqExecutions(req_id, exec_filter)
            
            self.logger.info("Requested executions")
            
        except Exception as e:
            self.logger.error(f"Error requesting executions: {e}")
    
    def _handle_next_valid_id(self, order_id: int):
        """Handle next valid order ID"""
        if order_id > self.next_order_id:
            self.next_order_id = order_id
        self.logger.debug(f"Next valid order ID: {order_id}")
    
    def _handle_order_status(self, order_id: int, status: str, filled: float, remaining: float,
                           avg_fill_price: float, perm_id: int, parent_id: int, last_fill_price: float,
                           client_id: int, why_held: str, mkt_cap_price: float):
        """Handle order status updates"""
        try:
            if order_id in self.orders:
                order_data = self.orders[order_id]
            else:
                # Create placeholder order data if not found
                order_data = IBOrderData(
                    order_id=order_id,
                    client_id=client_id,
                    perm_id=perm_id,
                    contract=Contract(),  # Placeholder
                    order=Order(),  # Placeholder
                    symbol="UNKNOWN",
                    action="UNKNOWN",
                    order_type="UNKNOWN",
                    total_quantity=Decimal(str(filled + remaining))
                )
                self.orders[order_id] = order_data
            
            # Update order data
            order_data.status = status
            order_data.filled_quantity = Decimal(str(filled))
            order_data.remaining_quantity = Decimal(str(remaining))
            order_data.perm_id = perm_id
            order_data.parent_id = parent_id if parent_id != 0 else None
            order_data.why_held = why_held if why_held else None
            order_data.updated_at = datetime.now()
            
            if avg_fill_price > 0:
                order_data.avg_fill_price = Decimal(str(avg_fill_price))
            if last_fill_price > 0:
                order_data.last_fill_price = Decimal(str(last_fill_price))
            if mkt_cap_price > 0:
                order_data.mkt_cap_price = Decimal(str(mkt_cap_price))
            
            # Remove from pending if not pending anymore
            if status not in [IBOrderStatus.PENDING_SUBMIT.value, IBOrderStatus.PENDING_CANCEL.value]:
                self.pending_orders.discard(order_id)
            
            self.logger.debug(f"Order {order_id} status: {status}, filled: {filled}, remaining: {remaining}")
            
            # Notify callbacks
            asyncio.create_task(self._notify_order_status_callbacks(order_data))
            
        except Exception as e:
            self.logger.error(f"Error handling order status: {e}")
    
    def _handle_open_order(self, order_id: int, contract: Contract, order: Order, order_state):
        """Handle open order updates"""
        try:
            if order_id in self.orders:
                order_data = self.orders[order_id]
                # Update contract and order details
                order_data.contract = contract
                order_data.order = order
                order_data.symbol = contract.symbol
                order_data.action = order.action
                order_data.order_type = order.orderType
                order_data.total_quantity = Decimal(str(order.totalQuantity))
            else:
                # Create new order data
                order_data = IBOrderData(
                    order_id=order_id,
                    client_id=order.clientId if hasattr(order, 'clientId') else 0,
                    perm_id=order.permId if hasattr(order, 'permId') else 0,
                    contract=contract,
                    order=order,
                    symbol=contract.symbol,
                    action=order.action,
                    order_type=order.orderType,
                    total_quantity=Decimal(str(order.totalQuantity)),
                    remaining_quantity=Decimal(str(order.totalQuantity))
                )
                self.orders[order_id] = order_data
            
            # Update order state information
            if hasattr(order_state, 'initMarginBefore'):
                order_data.init_margin_before = order_state.initMarginBefore
            if hasattr(order_state, 'maintMarginBefore'):
                order_data.maint_margin_before = order_state.maintMarginBefore
            if hasattr(order_state, 'equityWithLoanBefore'):
                order_data.equity_with_loan_before = order_state.equityWithLoanBefore
            if hasattr(order_state, 'initMarginChange'):
                order_data.init_margin_change = order_state.initMarginChange
            if hasattr(order_state, 'maintMarginChange'):
                order_data.maint_margin_change = order_state.maintMarginChange
            if hasattr(order_state, 'equityWithLoanChange'):
                order_data.equity_with_loan_change = order_state.equityWithLoanChange
            if hasattr(order_state, 'initMarginAfter'):
                order_data.init_margin_after = order_state.initMarginAfter
            if hasattr(order_state, 'maintMarginAfter'):
                order_data.maint_margin_after = order_state.maintMarginAfter
            if hasattr(order_state, 'equityWithLoanAfter'):
                order_data.equity_with_loan_after = order_state.equityWithLoanAfter
            if hasattr(order_state, 'commission'):
                order_data.commission = Decimal(str(order_state.commission)) if order_state.commission else None
            if hasattr(order_state, 'minCommission'):
                order_data.min_commission = Decimal(str(order_state.minCommission)) if order_state.minCommission else None
            if hasattr(order_state, 'maxCommission'):
                order_data.max_commission = Decimal(str(order_state.maxCommission)) if order_state.maxCommission else None
            if hasattr(order_state, 'commissionCurrency'):
                order_data.commission_currency = order_state.commissionCurrency
            if hasattr(order_state, 'warningText'):
                order_data.warning_text = order_state.warningText
            
            order_data.updated_at = datetime.now()
            
            self.logger.debug(f"Open order {order_id}: {contract.symbol} {order.action} {order.totalQuantity}")
            
        except Exception as e:
            self.logger.error(f"Error handling open order: {e}")
    
    def _handle_open_order_end(self):
        """Handle end of open orders"""
        self.logger.debug("Open orders request completed")
    
    def _handle_execution_details(self, req_id: int, contract: Contract, execution: Execution):
        """Handle execution details"""
        try:
            execution_data = IBOrderExecution(
                execution_id=execution.execId,
                order_id=execution.orderId,
                contract_id=contract.conId,
                symbol=contract.symbol,
                side=execution.side,
                shares=Decimal(str(execution.shares)),
                price=Decimal(str(execution.price)),
                perm_id=execution.permId,
                client_id=execution.clientId,
                exchange=execution.exchange,
                acct_number=execution.acctNumber,
                time=execution.time
            )
            
            # Store execution
            self.executions[execution.execId] = execution_data
            
            # Add to order's execution list
            if execution.orderId in self.orders:
                order_data = self.orders[execution.orderId]
                order_data.executions.append(execution_data)
                order_data.updated_at = datetime.now()
            
            self.logger.info(f"Execution: {execution.execId} - {execution.side} {execution.shares} {contract.symbol} @ {execution.price}")
            
            # Notify callbacks
            asyncio.create_task(self._notify_execution_callbacks(execution_data))
            
        except Exception as e:
            self.logger.error(f"Error handling execution details: {e}")
    
    def _handle_execution_details_end(self, req_id: int):
        """Handle end of execution details"""
        self.logger.debug("Execution details request completed")
    
    def _handle_commission_report(self, commission_report: CommissionReport):
        """Handle commission report"""
        try:
            exec_id = commission_report.execId
            
            # Store commission report
            self.commission_reports[exec_id] = commission_report
            
            # Update execution with commission
            if exec_id in self.executions:
                execution = self.executions[exec_id]
                execution.commission = Decimal(str(commission_report.commission))
                execution.realized_pnl = Decimal(str(commission_report.realizedPNL)) if commission_report.realizedPNL else None
                execution.yield_redemption_date = commission_report.yieldRedemptionDate
                
                # Update order data if available
                if execution.order_id in self.orders:
                    order_data = self.orders[execution.order_id]
                    order_data.commission = execution.commission
                    order_data.realized_pnl = execution.realized_pnl
                    order_data.updated_at = datetime.now()
            
            self.logger.info(f"Commission report: {exec_id} - Commission: {commission_report.commission}")
            
            # Notify callbacks
            asyncio.create_task(self._notify_commission_callbacks(commission_report))
            
        except Exception as e:
            self.logger.error(f"Error handling commission report: {e}")
    
    async def _notify_order_status_callbacks(self, order_data: IBOrderData):
        """Notify order status callbacks"""
        for callback in self.order_status_callbacks:
            try:
                await callback(order_data)
            except Exception as e:
                self.logger.error(f"Error in order status callback: {e}")
    
    async def _notify_execution_callbacks(self, execution: IBOrderExecution):
        """Notify execution callbacks"""
        for callback in self.execution_callbacks:
            try:
                await callback(execution)
            except Exception as e:
                self.logger.error(f"Error in execution callback: {e}")
    
    async def _notify_commission_callbacks(self, commission_report: CommissionReport):
        """Notify commission callbacks"""
        for callback in self.commission_callbacks:
            try:
                await callback(commission_report)
            except Exception as e:
                self.logger.error(f"Error in commission callback: {e}")
    
    def add_order_status_callback(self, callback: Callable):
        """Add order status callback"""
        self.order_status_callbacks.append(callback)
    
    def add_execution_callback(self, callback: Callable):
        """Add execution callback"""
        self.execution_callbacks.append(callback)
    
    def add_commission_callback(self, callback: Callable):
        """Add commission callback"""
        self.commission_callbacks.append(callback)
    
    def get_order(self, order_id: int) -> Optional[IBOrderData]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> Dict[int, IBOrderData]:
        """Get all orders"""
        return self.orders.copy()
    
    def get_open_orders(self) -> Dict[int, IBOrderData]:
        """Get open orders"""
        return {order_id: order_data for order_id, order_data in self.orders.items()
                if order_data.status not in [IBOrderStatus.FILLED.value, IBOrderStatus.CANCELLED.value, IBOrderStatus.API_CANCELLED.value]}
    
    def get_executions(self) -> Dict[str, IBOrderExecution]:
        """Get all executions"""
        return self.executions.copy()
    
    def get_order_executions(self, order_id: int) -> List[IBOrderExecution]:
        """Get executions for a specific order"""
        if order_id in self.orders:
            return self.orders[order_id].executions.copy()
        return []
    
    def cleanup(self):
        """Cleanup order manager"""
        self.orders.clear()
        self.executions.clear()
        self.commission_reports.clear()
        self.pending_orders.clear()
        self.logger.info("Order manager cleaned up")


# Global order manager instance
_ib_order_manager: Optional[IBOrderManager] = None

def get_ib_order_manager(ib_client) -> IBOrderManager:
    """Get or create the IB order manager singleton"""
    global _ib_order_manager
    
    if _ib_order_manager is None:
        _ib_order_manager = IBOrderManager(ib_client)
    
    return _ib_order_manager

def reset_ib_order_manager():
    """Reset the order manager singleton (for testing)"""
    global _ib_order_manager
    if _ib_order_manager:
        _ib_order_manager.cleanup()
    _ib_order_manager = None
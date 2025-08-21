#!/usr/bin/env python3
"""
Direct test of IB order placement to verify what works
"""

import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order


class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.orders = {}

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"ERROR: {reqId}, {errorCode}, {errorString}")

    def connectAck(self):
        print("Connected to IB Gateway")
        self.connected = True

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        print(f"Next valid order ID: {orderId}")
        self.nextOrderId = orderId

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print(f"Order {orderId}: status={status}, filled={filled}, remaining={remaining}")
        self.orders[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice
        }


def create_stock_contract(symbol):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    return contract


def create_market_order(action, quantity):
    order = Order()
    # Essential IB order attributes
    order.action = action
    order.totalQuantity = int(quantity)  
    order.orderType = "MKT"
    
    # FIX for EtradeOnly error - set deprecated attributes to empty strings
    order.eTradeOnly = ""
    order.firmQuoteOnly = ""
    
    return order


def main():
    app = IBApi()
    
    # Connect to IB Gateway
    print("Connecting to IB Gateway...")
    app.connect('127.0.0.1', 4002, clientId=3)  # Use client ID 3
    
    # Start message loop in separate thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    # Wait for connection
    time.sleep(2)
    
    if not app.connected:
        print("Failed to connect to IB Gateway")
        return
    
    # Wait for next valid order ID
    time.sleep(1)
    
    # Create and place a simple market order
    contract = create_stock_contract('SPY')
    order = create_market_order('BUY', 1)
    
    order_id = app.nextOrderId
    print(f"Placing order {order_id}: BUY 1 SPY MKT")
    
    app.placeOrder(order_id, contract, order)
    
    # Wait and check order status
    for i in range(10):
        time.sleep(2)
        if order_id in app.orders:
            status = app.orders[order_id]
            print(f"Order update: {status}")
            if status['status'] in ['Filled', 'Cancelled']:
                break
        else:
            print(f"Waiting for order status... ({i+1}/10)")
    
    print("Test complete")
    app.disconnect()


if __name__ == "__main__":
    main()
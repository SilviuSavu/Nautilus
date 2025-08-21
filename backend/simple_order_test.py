#!/usr/bin/env python3
"""
Simple test script to place a basic market order without advanced attributes
"""

import asyncio
import logging
from ibapi.order import Order
from ibapi.contract import Contract

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_contract(symbol: str) -> Contract:
    """Create the simplest possible stock contract"""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract

def create_simple_order(action: str, quantity: float) -> Order:
    """Create the simplest possible market order"""
    order = Order()
    order.action = action  # "BUY" or "SELL"
    order.orderType = "MKT"  # Market order
    order.totalQuantity = quantity
    order.tif = "DAY"  # Time in force
    
    # DO NOT SET any advanced attributes that could trigger EtradeOnly error:
    # - No blockOrder
    # - No sweepToFill  
    # - No hidden
    # - No discretionaryAmt
    # - No outsideRth
    # - No eTradeOnly
    
    return order

if __name__ == "__main__":
    # Test contract and order creation
    contract = create_simple_contract("AAPL")
    order = create_simple_order("BUY", 1)
    
    logger.info(f"Contract: {contract.symbol} {contract.secType} {contract.exchange} {contract.currency}")
    logger.info(f"Order: {order.action} {order.totalQuantity} {order.orderType} {order.tif}")
    logger.info("Simple order creation test complete")
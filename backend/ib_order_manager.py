"""
Nautilus IB Order Manager Stub
Provides compatibility stubs for Nautilus Trader migration.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from decimal import Decimal


@dataclass
class IBOrderData:
    """IB Order data structure - compatibility stub."""
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    order_type: str
    price: Optional[Decimal] = None
    status: str = "PENDING"
    filled_qty: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    timestamp: datetime = None


@dataclass 
class IBOrderExecution:
    """IB Order execution data - compatibility stub."""
    execution_id: str
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Optional[Decimal] = None
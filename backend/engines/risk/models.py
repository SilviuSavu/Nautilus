#!/usr/bin/env python3
"""
Risk Engine Models - Data classes and enums for risk management
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskLimitType(Enum):
    POSITION_SIZE = "position_size"
    PORTFOLIO_VALUE = "portfolio_value" 
    DAILY_LOSS = "daily_loss"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    VaR_LIMIT = "var_limit"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    SECTOR_EXPOSURE = "sector_exposure"
    COUNTRY_EXPOSURE = "country_exposure"
    CURRENCY_EXPOSURE = "currency_exposure"


class BreachSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimit:
    limit_id: str
    limit_type: RiskLimitType
    limit_value: float
    current_value: float
    threshold_warning: float = 0.8  # 80% of limit
    threshold_breach: float = 1.0   # 100% of limit
    enabled: bool = True
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class RiskBreach:
    breach_id: str
    limit_id: str
    breach_time: datetime
    severity: BreachSeverity
    breach_value: float
    limit_value: float
    breach_percentage: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    action_taken: Optional[str] = None
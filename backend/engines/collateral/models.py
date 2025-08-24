"""
Collateral Management Data Models
=================================

Core data structures for margin requirements, collateral tracking, and risk metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from decimal import Decimal


class AssetClass(Enum):
    """Asset classification for cross-margining purposes"""
    EQUITY = "equity"
    BOND = "bond"
    FX = "fx"
    COMMODITY = "commodity"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"


class MarginType(Enum):
    """Types of margin requirements"""
    INITIAL = "initial"
    VARIATION = "variation"
    MAINTENANCE = "maintenance"
    REGULATORY = "regulatory"


class AlertSeverity(Enum):
    """Margin alert severity levels"""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RecommendedAction(Enum):
    """Recommended actions for margin management"""
    MONITOR_CLOSELY = "monitor_closely"
    REDUCE_POSITIONS = "reduce_positions"
    ADD_COLLATERAL = "add_collateral"
    IMMEDIATE_ACTION_REQUIRED = "immediate_action_required"
    LIQUIDATE_POSITIONS = "liquidate_positions"


@dataclass
class Position:
    """Position data for margin calculations"""
    id: str
    symbol: str
    quantity: Decimal
    market_value: Decimal
    asset_class: AssetClass
    currency: str = "USD"
    sector: Optional[str] = None
    country: Optional[str] = None
    duration: Optional[Decimal] = None  # For bonds
    implied_volatility: Optional[Decimal] = None  # For options
    delta: Optional[Decimal] = None  # For derivatives
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position"""
        return abs(self.quantity * self.market_value)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0


@dataclass
class Portfolio:
    """Portfolio containing positions and cash"""
    id: str
    name: str
    positions: List[Position]
    available_cash: Decimal
    currency: str = "USD"
    leverage_ratio: Optional[Decimal] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def total_market_value(self) -> Decimal:
        """Calculate total market value of all positions"""
        return sum(pos.market_value * pos.quantity for pos in self.positions)
    
    @property
    def total_long_value(self) -> Decimal:
        """Calculate total value of long positions"""
        return sum(pos.market_value * pos.quantity for pos in self.positions if pos.is_long)
    
    @property
    def total_short_value(self) -> Decimal:
        """Calculate total value of short positions"""
        return sum(abs(pos.market_value * pos.quantity) for pos in self.positions if pos.is_short)
    
    @property
    def gross_exposure(self) -> Decimal:
        """Calculate gross exposure (long + short)"""
        return self.total_long_value + self.total_short_value
    
    @property
    def net_exposure(self) -> Decimal:
        """Calculate net exposure (long - short)"""
        return self.total_long_value - self.total_short_value


@dataclass
class MarginRequirement:
    """Comprehensive margin requirement calculation"""
    portfolio_id: str
    gross_initial_margin: Decimal
    cross_margin_offset: Decimal
    net_initial_margin: Decimal
    variation_margin: Decimal
    maintenance_margin: Decimal
    regulatory_margin: Decimal
    total_margin_requirement: Decimal
    margin_utilization: Decimal
    margin_excess: Decimal
    time_to_margin_call_minutes: Optional[int] = None
    stress_test_margin: Optional[Decimal] = None
    position_margins: Dict[str, Dict[str, Decimal]] = field(default_factory=dict)
    currency: str = "USD"
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def margin_utilization_percent(self) -> float:
        """Get margin utilization as percentage"""
        return float(self.margin_utilization * 100)
    
    @property
    def is_margin_call_risk(self) -> bool:
        """Check if margin call risk exists"""
        return self.margin_utilization > Decimal('0.8')  # 80% threshold
    
    @property
    def is_critical_margin_level(self) -> bool:
        """Check if margin level is critical"""
        return self.margin_utilization > Decimal('0.9')  # 90% threshold


@dataclass
class MarginAlert:
    """Margin monitoring alert"""
    portfolio_id: str
    severity: AlertSeverity
    message: str
    margin_utilization: Decimal
    time_to_margin_call_minutes: Optional[int] = None
    recommended_action: Optional[RecommendedAction] = None
    affected_positions: List[str] = field(default_factory=list)
    required_action_amount: Optional[Decimal] = None
    currency: str = "USD"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    
    def acknowledge(self):
        """Acknowledge the alert"""
        self.acknowledged = True
        self.acknowledged_at = datetime.now(timezone.utc)


@dataclass
class CrossMarginBenefit:
    """Cross-margining benefit calculation"""
    asset_class: AssetClass
    position_ids: List[str]
    correlation_coefficient: Decimal
    gross_margin: Decimal
    cross_margin_offset: Decimal
    net_margin: Decimal
    offset_percentage: Decimal
    calculation_method: str
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def capital_efficiency_improvement(self) -> float:
        """Calculate capital efficiency improvement percentage"""
        if self.gross_margin > 0:
            return float((self.cross_margin_offset / self.gross_margin) * 100)
        return 0.0


@dataclass
class RegulatoryCapitalRequirement:
    """Regulatory capital requirement calculation"""
    portfolio_id: str
    basel_iii_requirement: Decimal
    dodd_frank_requirement: Decimal
    emir_requirement: Decimal
    local_regulatory_requirement: Decimal
    total_regulatory_capital: Decimal
    capital_adequacy_ratio: Decimal
    regulatory_framework: str
    jurisdiction: str = "US"
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_compliant(self) -> bool:
        """Check if regulatory capital requirements are met"""
        return self.capital_adequacy_ratio >= Decimal('1.0')


@dataclass
class MarginStressTest:
    """Margin stress test results"""
    portfolio_id: str
    scenario_name: str
    base_margin: Decimal
    stressed_margin: Decimal
    margin_increase: Decimal
    margin_increase_percent: Decimal
    positions_at_risk: List[str]
    estimated_liquidation_value: Decimal
    time_to_liquidation_minutes: Optional[int] = None
    stress_factors: Dict[str, Decimal] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def passes_stress_test(self) -> bool:
        """Check if portfolio passes stress test"""
        # Stress test passes if margin increase is less than 200%
        return self.margin_increase_percent < Decimal('2.0')


@dataclass
class CollateralMovement:
    """Collateral movement tracking"""
    id: str
    portfolio_id: str
    movement_type: str  # "deposit", "withdrawal", "transfer"
    amount: Decimal
    currency: str
    counterparty: Optional[str] = None
    settlement_date: Optional[datetime] = None
    status: str = "pending"  # "pending", "settled", "failed"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settled_at: Optional[datetime] = None
    reference_id: Optional[str] = None
    
    @property
    def is_settled(self) -> bool:
        """Check if collateral movement is settled"""
        return self.status == "settled"
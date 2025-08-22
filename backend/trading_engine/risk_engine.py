"""
Real-Time Risk Engine
====================

Comprehensive risk management system with real-time monitoring, limits enforcement,
and advanced risk analytics for institutional trading.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
import math

from .order_management import Order, OrderSide
from .position_keeper import Position

logger = logging.getLogger(__name__)


class RiskViolationType(Enum):
    """Types of risk violations."""
    POSITION_LIMIT = "position_limit"
    NOTIONAL_LIMIT = "notional_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VAR_LIMIT = "var_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    INTRADAY_LIMIT = "intraday_limit"


@dataclass
class RiskLimits:
    """Portfolio risk limits configuration."""
    portfolio_id: str
    
    # Position limits
    max_position_size: float = 1000000.0  # Max position value
    max_portfolio_concentration: float = 0.20  # Max 20% in single position
    max_sector_concentration: float = 0.30  # Max 30% in single sector
    
    # Loss limits
    max_daily_loss: float = 50000.0  # Max daily loss
    max_drawdown_percent: float = 0.10  # Max 10% drawdown
    
    # Leverage limits
    max_gross_leverage: float = 2.0  # Max 2x gross leverage
    max_net_leverage: float = 1.5   # Max 1.5x net leverage
    
    # Risk metrics limits
    max_var_99: float = 100000.0  # Max 99% VaR
    max_expected_shortfall: float = 150000.0  # Max expected shortfall
    
    # Intraday limits
    max_intraday_turnover: float = 5000000.0  # Max intraday turnover
    max_order_size: float = 100000.0  # Max single order size
    
    # Time-based limits
    trading_hours_start: str = "09:30"  # Trading hours start
    trading_hours_end: str = "16:00"    # Trading hours end
    
    # Enable/disable flags
    enforce_position_limits: bool = True
    enforce_loss_limits: bool = True
    enforce_leverage_limits: bool = True
    enforce_concentration_limits: bool = True
    enforce_var_limits: bool = True
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskMetrics:
    """Real-time risk metrics for a portfolio."""
    portfolio_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Position metrics
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    
    # Leverage metrics
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    
    # Concentration metrics
    largest_position_percent: float = 0.0
    top_5_concentration: float = 0.0
    sector_concentrations: Dict[str, float] = field(default_factory=dict)
    
    # P&L metrics
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Risk metrics
    var_99_1d: float = 0.0  # 1-day 99% VaR
    expected_shortfall: float = 0.0
    beta_market: float = 0.0
    
    # Drawdown metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_portfolio_value: float = 0.0
    
    # Intraday metrics
    intraday_turnover: float = 0.0
    orders_count: int = 0
    rejected_orders: int = 0


@dataclass
class RiskViolation:
    """Risk limit violation event."""
    violation_type: RiskViolationType
    portfolio_id: str
    description: str
    current_value: float
    limit_value: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskRule(ABC):
    """Abstract base class for risk rules."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    async def check(self, order: Order, portfolio_id: str, positions: Dict[str, Position], 
                   metrics: RiskMetrics, limits: RiskLimits) -> Optional[RiskViolation]:
        """Check if order violates this risk rule."""
        pass


class PositionSizeRule(RiskRule):
    """Rule to check position size limits."""
    
    def __init__(self):
        super().__init__("Position Size Limit")
    
    async def check(self, order: Order, portfolio_id: str, positions: Dict[str, Position], 
                   metrics: RiskMetrics, limits: RiskLimits) -> Optional[RiskViolation]:
        
        if not limits.enforce_position_limits:
            return None
        
        # Calculate new position size after order
        current_position = positions.get(order.symbol, Position(order.symbol, 0.0, 0.0))
        
        if order.side == OrderSide.BUY:
            new_quantity = current_position.quantity + order.quantity
        else:
            new_quantity = current_position.quantity - order.quantity
        
        # Estimate position value (using order price or current market price)
        price = order.price if order.price else current_position.average_price
        new_position_value = abs(new_quantity * price)
        
        if new_position_value > limits.max_position_size:
            return RiskViolation(
                violation_type=RiskViolationType.POSITION_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Position size {new_position_value:,.2f} exceeds limit {limits.max_position_size:,.2f} for {order.symbol}",
                current_value=new_position_value,
                limit_value=limits.max_position_size,
                severity="HIGH" if new_position_value > limits.max_position_size * 1.5 else "MEDIUM"
            )
        
        return None


class LeverageRule(RiskRule):
    """Rule to check leverage limits."""
    
    def __init__(self):
        super().__init__("Leverage Limit")
    
    async def check(self, order: Order, portfolio_id: str, positions: Dict[str, Position], 
                   metrics: RiskMetrics, limits: RiskLimits) -> Optional[RiskViolation]:
        
        if not limits.enforce_leverage_limits:
            return None
        
        # Estimate impact of order on leverage
        order_value = order.quantity * (order.price or 100.0)  # Default price if not specified
        
        estimated_gross_exposure = metrics.gross_exposure + order_value
        
        # Assume portfolio value (would be calculated from positions and cash)
        portfolio_value = max(metrics.gross_exposure * 0.5, 1000000.0)  # Simplified assumption
        
        estimated_gross_leverage = estimated_gross_exposure / portfolio_value
        
        if estimated_gross_leverage > limits.max_gross_leverage:
            return RiskViolation(
                violation_type=RiskViolationType.LEVERAGE_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Gross leverage {estimated_gross_leverage:.2f} would exceed limit {limits.max_gross_leverage:.2f}",
                current_value=estimated_gross_leverage,
                limit_value=limits.max_gross_leverage,
                severity="CRITICAL" if estimated_gross_leverage > limits.max_gross_leverage * 1.2 else "HIGH"
            )
        
        return None


class DailyLossRule(RiskRule):
    """Rule to check daily loss limits."""
    
    def __init__(self):
        super().__init__("Daily Loss Limit")
    
    async def check(self, order: Order, portfolio_id: str, positions: Dict[str, Position], 
                   metrics: RiskMetrics, limits: RiskLimits) -> Optional[RiskViolation]:
        
        if not limits.enforce_loss_limits:
            return None
        
        if metrics.daily_pnl < -limits.max_daily_loss:
            return RiskViolation(
                violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Daily loss {abs(metrics.daily_pnl):,.2f} exceeds limit {limits.max_daily_loss:,.2f}",
                current_value=abs(metrics.daily_pnl),
                limit_value=limits.max_daily_loss,
                severity="CRITICAL"
            )
        
        return None


class ConcentrationRule(RiskRule):
    """Rule to check concentration limits."""
    
    def __init__(self):
        super().__init__("Concentration Limit")
    
    async def check(self, order: Order, portfolio_id: str, positions: Dict[str, Position], 
                   metrics: RiskMetrics, limits: RiskLimits) -> Optional[RiskViolation]:
        
        if not limits.enforce_concentration_limits:
            return None
        
        # Check if largest single position would exceed concentration limit
        if metrics.largest_position_percent > limits.max_portfolio_concentration * 100:
            return RiskViolation(
                violation_type=RiskViolationType.CONCENTRATION_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Position concentration {metrics.largest_position_percent:.1f}% exceeds limit {limits.max_portfolio_concentration*100:.1f}%",
                current_value=metrics.largest_position_percent,
                limit_value=limits.max_portfolio_concentration * 100,
                severity="MEDIUM"
            )
        
        return None


class RealTimeRiskEngine:
    """
    Real-time risk management engine with comprehensive monitoring,
    limits enforcement, and violation reporting.
    """
    
    def __init__(self):
        self.portfolio_limits: Dict[str, RiskLimits] = {}
        self.portfolio_metrics: Dict[str, RiskMetrics] = {}
        self.risk_rules: List[RiskRule] = []
        self.violation_callbacks: List[Callable[[RiskViolation], Any]] = []
        self.position_keeper = None
        self.monitoring_enabled = True
        
        # Initialize default risk rules
        self._init_default_rules()
        
    def _init_default_rules(self):
        """Initialize default risk rules."""
        self.risk_rules = [
            PositionSizeRule(),
            LeverageRule(),
            DailyLossRule(),
            ConcentrationRule()
        ]
    
    def set_position_keeper(self, position_keeper):
        """Set position keeper for real-time position data."""
        self.position_keeper = position_keeper
    
    def add_violation_callback(self, callback: Callable[[RiskViolation], Any]):
        """Add callback for risk violations."""
        self.violation_callbacks.append(callback)
    
    def set_portfolio_limits(self, portfolio_id: str, limits: RiskLimits):
        """Set risk limits for a portfolio."""
        self.portfolio_limits[portfolio_id] = limits
        logger.info(f"Updated risk limits for portfolio {portfolio_id}")
    
    def get_portfolio_limits(self, portfolio_id: str) -> Optional[RiskLimits]:
        """Get risk limits for a portfolio."""
        return self.portfolio_limits.get(portfolio_id)
    
    def add_risk_rule(self, rule: RiskRule):
        """Add custom risk rule."""
        self.risk_rules.append(rule)
        logger.info(f"Added risk rule: {rule.name}")
    
    async def check_pre_trade_risk(self, order: Order, portfolio_id: str) -> bool:
        """
        Check pre-trade risk for an order.
        
        Args:
            order: Order to check
            portfolio_id: Portfolio ID
            
        Returns:
            bool: True if order passes all risk checks
        """
        try:
            # Get portfolio limits
            limits = self.portfolio_limits.get(portfolio_id)
            if not limits:
                logger.warning(f"No risk limits set for portfolio {portfolio_id}")
                return True  # Allow if no limits configured
            
            # Get current positions
            positions = {}
            if self.position_keeper:
                positions = await self.position_keeper.get_positions(portfolio_id)
            
            # Get current metrics
            metrics = await self.calculate_portfolio_metrics(portfolio_id, positions)
            
            # Check all risk rules
            for rule in self.risk_rules:
                if not rule.enabled:
                    continue
                
                violation = await rule.check(order, portfolio_id, positions, metrics, limits)
                if violation:
                    logger.warning(f"Risk violation: {violation.description}")
                    await self._notify_violation(violation)
                    return False
            
            logger.debug(f"Order {order.id} passed all risk checks")
            return True
            
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return False  # Fail safe - reject order on error
    
    async def calculate_portfolio_metrics(self, portfolio_id: str, 
                                        positions: Dict[str, Position] = None) -> RiskMetrics:
        """Calculate real-time portfolio risk metrics."""
        
        if positions is None and self.position_keeper:
            positions = await self.position_keeper.get_positions(portfolio_id)
        elif positions is None:
            positions = {}
        
        metrics = RiskMetrics(portfolio_id=portfolio_id)
        
        if not positions:
            self.portfolio_metrics[portfolio_id] = metrics
            return metrics
        
        # Calculate exposure metrics
        long_exposure = 0.0
        short_exposure = 0.0
        position_values = []
        
        for position in positions.values():
            position_value = position.quantity * position.average_price
            
            if position.quantity > 0:
                long_exposure += position_value
            else:
                short_exposure += abs(position_value)
            
            position_values.append(abs(position_value))
        
        metrics.long_exposure = long_exposure
        metrics.short_exposure = short_exposure
        metrics.gross_exposure = long_exposure + short_exposure
        metrics.net_exposure = long_exposure - short_exposure
        
        # Calculate leverage (simplified - would need portfolio cash balance)
        portfolio_value = max(metrics.gross_exposure * 0.5, 1000000.0)  # Simplified
        metrics.gross_leverage = metrics.gross_exposure / portfolio_value
        metrics.net_leverage = abs(metrics.net_exposure) / portfolio_value
        
        # Calculate concentration metrics
        if position_values:
            total_exposure = sum(position_values)
            if total_exposure > 0:
                metrics.largest_position_percent = (max(position_values) / total_exposure) * 100
                
                # Top 5 concentration
                top_5 = sorted(position_values, reverse=True)[:5]
                metrics.top_5_concentration = (sum(top_5) / total_exposure) * 100
        
        # Calculate P&L metrics (simplified)
        total_unrealized_pnl = 0.0
        for position in positions.values():
            # Would need current market prices to calculate unrealized P&L
            # This is a simplified calculation
            unrealized = position.unrealized_pnl if hasattr(position, 'unrealized_pnl') else 0.0
            total_unrealized_pnl += unrealized
        
        metrics.unrealized_pnl = total_unrealized_pnl
        metrics.total_pnl = metrics.unrealized_pnl + metrics.realized_pnl
        
        # Store metrics
        self.portfolio_metrics[portfolio_id] = metrics
        
        return metrics
    
    async def _notify_violation(self, violation: RiskViolation):
        """Notify all callbacks of risk violation."""
        for callback in self.violation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(violation)
                else:
                    callback(violation)
            except Exception as e:
                logger.error(f"Risk violation callback failed: {e}")
    
    def get_portfolio_metrics(self, portfolio_id: str) -> Optional[RiskMetrics]:
        """Get current portfolio metrics."""
        return self.portfolio_metrics.get(portfolio_id)
    
    async def emergency_stop_trading(self, portfolio_id: str, reason: str):
        """Emergency stop all trading for a portfolio."""
        logger.critical(f"EMERGENCY STOP for portfolio {portfolio_id}: {reason}")
        
        # Create critical violation
        violation = RiskViolation(
            violation_type=RiskViolationType.DAILY_LOSS_LIMIT,  # Use as emergency type
            portfolio_id=portfolio_id,
            description=f"EMERGENCY STOP: {reason}",
            current_value=0.0,
            limit_value=0.0,
            severity="CRITICAL",
            metadata={"emergency_stop": True}
        )
        
        await self._notify_violation(violation)
    
    async def start_monitoring(self):
        """Start real-time risk monitoring."""
        if self.monitoring_enabled:
            asyncio.create_task(self._monitor_risk())
    
    async def _monitor_risk(self):
        """Monitor portfolio risk metrics."""
        while self.monitoring_enabled:
            try:
                for portfolio_id in self.portfolio_limits.keys():
                    # Update metrics
                    await self.calculate_portfolio_metrics(portfolio_id)
                    
                    # Check for violations in current metrics
                    metrics = self.portfolio_metrics.get(portfolio_id)
                    limits = self.portfolio_limits.get(portfolio_id)
                    
                    if metrics and limits:
                        await self._check_portfolio_violations(portfolio_id, metrics, limits)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _check_portfolio_violations(self, portfolio_id: str, metrics: RiskMetrics, limits: RiskLimits):
        """Check for risk violations in current portfolio state."""
        
        # Check daily loss limit
        if limits.enforce_loss_limits and metrics.daily_pnl < -limits.max_daily_loss:
            violation = RiskViolation(
                violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Daily loss limit exceeded: {abs(metrics.daily_pnl):,.2f}",
                current_value=abs(metrics.daily_pnl),
                limit_value=limits.max_daily_loss,
                severity="CRITICAL"
            )
            await self._notify_violation(violation)
        
        # Check leverage limits
        if limits.enforce_leverage_limits and metrics.gross_leverage > limits.max_gross_leverage:
            violation = RiskViolation(
                violation_type=RiskViolationType.LEVERAGE_LIMIT,
                portfolio_id=portfolio_id,
                description=f"Gross leverage limit exceeded: {metrics.gross_leverage:.2f}",
                current_value=metrics.gross_leverage,
                limit_value=limits.max_gross_leverage,
                severity="HIGH"
            )
            await self._notify_violation(violation)
    
    def get_risk_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive risk summary for a portfolio."""
        metrics = self.portfolio_metrics.get(portfolio_id)
        limits = self.portfolio_limits.get(portfolio_id)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        summary = {
            "portfolio_id": portfolio_id,
            "timestamp": metrics.timestamp.isoformat(),
            "exposure": {
                "gross": metrics.gross_exposure,
                "net": metrics.net_exposure,
                "long": metrics.long_exposure,
                "short": metrics.short_exposure
            },
            "leverage": {
                "gross": metrics.gross_leverage,
                "net": metrics.net_leverage
            },
            "concentration": {
                "largest_position_percent": metrics.largest_position_percent,
                "top_5_concentration": metrics.top_5_concentration
            },
            "pnl": {
                "daily": metrics.daily_pnl,
                "unrealized": metrics.unrealized_pnl,
                "total": metrics.total_pnl
            }
        }
        
        if limits:
            summary["limits"] = {
                "max_position_size": limits.max_position_size,
                "max_daily_loss": limits.max_daily_loss,
                "max_gross_leverage": limits.max_gross_leverage,
                "max_concentration": limits.max_portfolio_concentration * 100
            }
            
            # Calculate utilization percentages
            summary["utilization"] = {
                "leverage": (metrics.gross_leverage / limits.max_gross_leverage) * 100 if limits.max_gross_leverage > 0 else 0,
                "daily_loss": (abs(metrics.daily_pnl) / limits.max_daily_loss) * 100 if limits.max_daily_loss > 0 else 0,
                "concentration": (metrics.largest_position_percent / (limits.max_portfolio_concentration * 100)) * 100 if limits.max_portfolio_concentration > 0 else 0
            }
        
        return summary
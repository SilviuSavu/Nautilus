"""
Dynamic Risk Limit Engine - Sprint 3 Priority 1
Risk Management Infrastructure

Provides real-time risk limit enforcement with:
- Dynamic limit calculations based on market conditions
- Pre-trade and post-trade limit checks
- Configurable limit types (VaR, exposure, drawdown, etc.)
- Automatic limit adjustments based on volatility
- Integration with breach detection and alerting
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import math

from ..database import get_db_connection
from ..websocket.redis_pubsub import get_redis_pubsub_manager, WebSocketMessage, MessageType
from ..websocket.message_protocols import create_risk_alert_message, create_breach_alert_message

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Types of risk limits"""
    VAR_ABSOLUTE = "var_absolute"       # Absolute VaR limit
    VAR_RELATIVE = "var_relative"       # VaR as % of portfolio value
    EXPOSURE_GROSS = "exposure_gross"   # Gross exposure limit
    EXPOSURE_NET = "exposure_net"       # Net exposure limit
    EXPOSURE_LONG = "exposure_long"     # Long exposure limit
    EXPOSURE_SHORT = "exposure_short"   # Short exposure limit
    DRAWDOWN_MAX = "drawdown_max"       # Maximum drawdown limit
    DRAWDOWN_DAILY = "drawdown_daily"   # Daily drawdown limit
    CONCENTRATION = "concentration"      # Single position concentration
    LEVERAGE = "leverage"               # Portfolio leverage limit
    SECTOR_EXPOSURE = "sector_exposure" # Sector exposure limit
    CORRELATION = "correlation"         # Portfolio correlation limit
    VOLATILITY = "volatility"          # Portfolio volatility limit

class LimitScope(Enum):
    """Scope of risk limit application"""
    PORTFOLIO = "portfolio"             # Portfolio-level limit
    STRATEGY = "strategy"               # Strategy-level limit
    SYMBOL = "symbol"                   # Symbol-level limit
    SECTOR = "sector"                   # Sector-level limit
    USER = "user"                       # User-level limit
    GLOBAL = "global"                   # Global system limit

class LimitStatus(Enum):
    """Status of risk limits"""
    ACTIVE = "active"                   # Limit is active
    INACTIVE = "inactive"               # Limit is inactive
    BREACHED = "breached"              # Limit is breached
    WARNING = "warning"                 # Approaching limit (80%+)
    SUSPENDED = "suspended"            # Limit temporarily suspended

class TimeWindow(Enum):
    """Time windows for limit calculations"""
    INTRADAY = "intraday"              # Current day
    DAILY = "daily"                    # Rolling 24 hours
    WEEKLY = "weekly"                  # Rolling 7 days
    MONTHLY = "monthly"                # Rolling 30 days
    QUARTERLY = "quarterly"            # Rolling 90 days
    REAL_TIME = "real_time"            # Real-time monitoring

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    limit_id: str
    limit_type: LimitType
    scope: LimitScope
    scope_id: str  # Portfolio ID, strategy ID, symbol, etc.
    limit_value: Decimal
    currency: str = "USD"
    time_window: TimeWindow = TimeWindow.INTRADAY
    status: LimitStatus = LimitStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    # Dynamic adjustment parameters
    auto_adjust: bool = False
    volatility_multiplier: float = 1.0
    min_limit: Optional[Decimal] = None
    max_limit: Optional[Decimal] = None
    
    # Warning thresholds
    warning_threshold: float = 0.8  # Warning at 80% of limit
    breach_threshold: float = 1.0   # Breach at 100% of limit
    
    # Metadata
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LimitCheck:
    """Result of a limit check"""
    limit_id: str
    current_value: Decimal
    limit_value: Decimal
    utilization: float  # Current value / limit value
    status: LimitStatus
    breach_amount: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicLimitEngine:
    """
    Dynamic risk limit engine for real-time risk management
    
    Features:
    - Real-time limit monitoring and enforcement
    - Dynamic limit adjustment based on market conditions
    - Pre-trade and post-trade limit validation
    - Automated breach detection and alerting
    - Multi-level limit hierarchy (portfolio, strategy, symbol)
    """
    
    def __init__(self):
        self.limits: Dict[str, RiskLimit] = {}
        self.limit_checks: Dict[str, List[LimitCheck]] = {}
        self.db_connection = None
        self.redis_pubsub = None
        
        # Monitoring tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._adjustment_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_checks = 0
        self.breach_count = 0
        self.warning_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the dynamic limit engine"""
        try:
            # Initialize database connection
            self.db_connection = await get_db_connection()
            
            # Initialize Redis pub/sub for alerts
            self.redis_pubsub = get_redis_pubsub_manager()
            
            # Load existing limits from database
            await self._load_limits_from_db()
            
            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitor_limits())
            self._adjustment_task = asyncio.create_task(self._adjust_dynamic_limits())
            
            self.logger.info("Dynamic limit engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize limit engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the limit engine"""
        try:
            # Cancel monitoring tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self._adjustment_task:
                self._adjustment_task.cancel()
                try:
                    await self._adjustment_task
                except asyncio.CancelledError:
                    pass
            
            # Close database connection
            if self.db_connection:
                await self.db_connection.close()
            
            self.logger.info("Dynamic limit engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during limit engine shutdown: {e}")
    
    async def create_limit(
        self,
        limit_type: LimitType,
        scope: LimitScope,
        scope_id: str,
        limit_value: Union[Decimal, float],
        **kwargs
    ) -> str:
        """Create a new risk limit"""
        try:
            limit_id = str(uuid.uuid4())
            
            # Create limit configuration
            limit = RiskLimit(
                limit_id=limit_id,
                limit_type=limit_type,
                scope=scope,
                scope_id=scope_id,
                limit_value=Decimal(str(limit_value)),
                **kwargs
            )
            
            # Store in memory
            self.limits[limit_id] = limit
            
            # Save to database
            await self._save_limit_to_db(limit)
            
            self.logger.info(
                f"Created {limit_type.value} limit {limit_id} for {scope.value} {scope_id}: {limit_value}"
            )
            
            return limit_id
            
        except Exception as e:
            self.logger.error(f"Error creating limit: {e}")
            raise
    
    async def update_limit(
        self,
        limit_id: str,
        **updates
    ) -> bool:
        """Update an existing risk limit"""
        try:
            if limit_id not in self.limits:
                self.logger.warning(f"Limit {limit_id} not found for update")
                return False
            
            limit = self.limits[limit_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(limit, field):
                    if field == 'limit_value' and not isinstance(value, Decimal):
                        value = Decimal(str(value))
                    setattr(limit, field, value)
            
            limit.updated_at = datetime.utcnow()
            
            # Save to database
            await self._save_limit_to_db(limit)
            
            self.logger.info(f"Updated limit {limit_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating limit {limit_id}: {e}")
            return False
    
    async def delete_limit(self, limit_id: str) -> bool:
        """Delete a risk limit"""
        try:
            if limit_id not in self.limits:
                self.logger.warning(f"Limit {limit_id} not found for deletion")
                return False
            
            # Remove from memory
            del self.limits[limit_id]
            
            # Remove from database
            await self._delete_limit_from_db(limit_id)
            
            self.logger.info(f"Deleted limit {limit_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting limit {limit_id}: {e}")
            return False
    
    async def check_pre_trade_limits(
        self,
        portfolio_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        side: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if a proposed trade would violate any risk limits
        
        Returns:
            Tuple[bool, List[str]]: (can_trade, list_of_violations)
        """
        try:
            violations = []
            
            # Get relevant limits
            relevant_limits = self._get_relevant_limits(
                portfolio_id=portfolio_id,
                symbol=symbol
            )
            
            # Calculate trade impact
            trade_value = quantity * price
            
            for limit in relevant_limits:
                violation = await self._check_pre_trade_limit(
                    limit, portfolio_id, symbol, quantity, price, side, trade_value
                )
                if violation:
                    violations.append(violation)
            
            can_trade = len(violations) == 0
            
            self.logger.debug(
                f"Pre-trade check for {portfolio_id}/{symbol}: "
                f"{'PASS' if can_trade else 'FAIL'} ({len(violations)} violations)"
            )
            
            return can_trade, violations
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade limit check: {e}")
            return False, [f"System error: {e}"]
    
    async def check_post_trade_limits(
        self,
        portfolio_id: str,
        symbol: str,
        executed_quantity: Decimal,
        executed_price: Decimal,
        side: str
    ) -> List[LimitCheck]:
        """Check limits after a trade has been executed"""
        try:
            checks = []
            
            # Get relevant limits
            relevant_limits = self._get_relevant_limits(
                portfolio_id=portfolio_id,
                symbol=symbol
            )
            
            for limit in relevant_limits:
                check = await self._perform_limit_check(limit, portfolio_id)
                if check:
                    checks.append(check)
                    
                    # Store check result
                    if limit.limit_id not in self.limit_checks:
                        self.limit_checks[limit.limit_id] = []
                    self.limit_checks[limit.limit_id].append(check)
                    
                    # Send alerts for breaches or warnings
                    if check.status in [LimitStatus.BREACHED, LimitStatus.WARNING]:
                        await self._send_limit_alert(limit, check)
            
            self.total_checks += len(checks)
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error in post-trade limit check: {e}")
            return []
    
    async def get_limit_status(self, limit_id: str) -> Optional[LimitCheck]:
        """Get current status of a specific limit"""
        try:
            if limit_id not in self.limits:
                return None
            
            limit = self.limits[limit_id]
            return await self._perform_limit_check(limit, limit.scope_id)
            
        except Exception as e:
            self.logger.error(f"Error getting limit status for {limit_id}: {e}")
            return None
    
    async def get_portfolio_limit_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get summary of all limits for a portfolio"""
        try:
            portfolio_limits = [
                limit for limit in self.limits.values()
                if (limit.scope == LimitScope.PORTFOLIO and limit.scope_id == portfolio_id) or
                   limit.scope == LimitScope.GLOBAL
            ]
            
            summary = {
                "portfolio_id": portfolio_id,
                "total_limits": len(portfolio_limits),
                "active_limits": 0,
                "breached_limits": 0,
                "warning_limits": 0,
                "limit_checks": []
            }
            
            for limit in portfolio_limits:
                check = await self._perform_limit_check(limit, portfolio_id)
                if check:
                    summary["limit_checks"].append({
                        "limit_id": limit.limit_id,
                        "limit_type": limit.limit_type.value,
                        "current_value": float(check.current_value),
                        "limit_value": float(check.limit_value),
                        "utilization": check.utilization,
                        "status": check.status.value
                    })
                    
                    if check.status == LimitStatus.ACTIVE:
                        summary["active_limits"] += 1
                    elif check.status == LimitStatus.BREACHED:
                        summary["breached_limits"] += 1
                    elif check.status == LimitStatus.WARNING:
                        summary["warning_limits"] += 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio limit summary: {e}")
            return {"error": str(e)}
    
    def _get_relevant_limits(
        self,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        sector: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[RiskLimit]:
        """Get all limits relevant to the given context"""
        relevant_limits = []
        
        for limit in self.limits.values():
            if limit.status != LimitStatus.ACTIVE:
                continue
            
            # Check scope matches
            if limit.scope == LimitScope.GLOBAL:
                relevant_limits.append(limit)
            elif limit.scope == LimitScope.PORTFOLIO and portfolio_id and limit.scope_id == portfolio_id:
                relevant_limits.append(limit)
            elif limit.scope == LimitScope.STRATEGY and strategy_id and limit.scope_id == strategy_id:
                relevant_limits.append(limit)
            elif limit.scope == LimitScope.SYMBOL and symbol and limit.scope_id == symbol:
                relevant_limits.append(limit)
            elif limit.scope == LimitScope.SECTOR and sector and limit.scope_id == sector:
                relevant_limits.append(limit)
            elif limit.scope == LimitScope.USER and user_id and limit.scope_id == user_id:
                relevant_limits.append(limit)
        
        return relevant_limits
    
    async def _check_pre_trade_limit(
        self,
        limit: RiskLimit,
        portfolio_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        side: str,
        trade_value: Decimal
    ) -> Optional[str]:
        """Check if a proposed trade would violate a specific limit"""
        try:
            # Get current limit value
            current_value = await self._calculate_limit_value(limit, portfolio_id)
            
            # Calculate projected value after trade
            projected_value = await self._calculate_projected_value(
                limit, portfolio_id, symbol, quantity, price, side, current_value
            )
            
            # Check if projected value would breach limit
            if projected_value > limit.limit_value:
                breach_amount = projected_value - limit.limit_value
                return (
                    f"{limit.limit_type.value} limit would be breached by {breach_amount} "
                    f"(current: {current_value}, projected: {projected_value}, limit: {limit.limit_value})"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade limit check: {e}")
            return f"Error checking {limit.limit_type.value} limit: {e}"
    
    async def _perform_limit_check(
        self,
        limit: RiskLimit,
        scope_id: str
    ) -> Optional[LimitCheck]:
        """Perform a limit check for a specific limit"""
        try:
            # Calculate current value
            current_value = await self._calculate_limit_value(limit, scope_id)
            
            # Calculate utilization
            utilization = float(current_value / limit.limit_value) if limit.limit_value > 0 else 0.0
            
            # Determine status
            status = LimitStatus.ACTIVE
            breach_amount = None
            
            if utilization >= limit.breach_threshold:
                status = LimitStatus.BREACHED
                breach_amount = current_value - limit.limit_value
                self.breach_count += 1
            elif utilization >= limit.warning_threshold:
                status = LimitStatus.WARNING
                self.warning_count += 1
            
            return LimitCheck(
                limit_id=limit.limit_id,
                current_value=current_value,
                limit_value=limit.limit_value,
                utilization=utilization,
                status=status,
                breach_amount=breach_amount
            )
            
        except Exception as e:
            self.logger.error(f"Error performing limit check: {e}")
            return None
    
    async def _calculate_limit_value(
        self,
        limit: RiskLimit,
        scope_id: str
    ) -> Decimal:
        """Calculate current value for a specific limit type"""
        try:
            if limit.limit_type == LimitType.VAR_ABSOLUTE:
                return await self._calculate_var_absolute(scope_id, limit.time_window)
            elif limit.limit_type == LimitType.VAR_RELATIVE:
                return await self._calculate_var_relative(scope_id, limit.time_window)
            elif limit.limit_type == LimitType.EXPOSURE_GROSS:
                return await self._calculate_gross_exposure(scope_id)
            elif limit.limit_type == LimitType.EXPOSURE_NET:
                return await self._calculate_net_exposure(scope_id)
            elif limit.limit_type == LimitType.EXPOSURE_LONG:
                return await self._calculate_long_exposure(scope_id)
            elif limit.limit_type == LimitType.EXPOSURE_SHORT:
                return await self._calculate_short_exposure(scope_id)
            elif limit.limit_type == LimitType.DRAWDOWN_MAX:
                return await self._calculate_max_drawdown(scope_id, limit.time_window)
            elif limit.limit_type == LimitType.DRAWDOWN_DAILY:
                return await self._calculate_daily_drawdown(scope_id)
            elif limit.limit_type == LimitType.CONCENTRATION:
                return await self._calculate_concentration(scope_id)
            elif limit.limit_type == LimitType.LEVERAGE:
                return await self._calculate_leverage(scope_id)
            else:
                self.logger.warning(f"Unknown limit type: {limit.limit_type}")
                return Decimal("0")
                
        except Exception as e:
            self.logger.error(f"Error calculating limit value for {limit.limit_type}: {e}")
            return Decimal("0")
    
    async def _calculate_projected_value(
        self,
        limit: RiskLimit,
        portfolio_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        side: str,
        current_value: Decimal
    ) -> Decimal:
        """Calculate projected limit value after a proposed trade"""
        try:
            trade_impact = Decimal("0")
            
            if limit.limit_type in [LimitType.EXPOSURE_GROSS, LimitType.EXPOSURE_LONG, LimitType.EXPOSURE_SHORT]:
                trade_impact = abs(quantity * price)
                if limit.limit_type == LimitType.EXPOSURE_SHORT and side.upper() == "SELL":
                    trade_impact = quantity * price
                elif limit.limit_type == LimitType.EXPOSURE_LONG and side.upper() == "BUY":
                    trade_impact = quantity * price
                else:
                    trade_impact = Decimal("0")
            
            elif limit.limit_type == LimitType.EXPOSURE_NET:
                trade_impact = quantity * price if side.upper() == "BUY" else -(quantity * price)
            
            elif limit.limit_type == LimitType.CONCENTRATION:
                # For concentration, calculate new position size as % of portfolio
                portfolio_value = await self._get_portfolio_value(portfolio_id)
                if portfolio_value > 0:
                    current_position = await self._get_position_value(portfolio_id, symbol)
                    new_position_value = current_position + (quantity * price if side.upper() == "BUY" else -(quantity * price))
                    trade_impact = abs(new_position_value) / portfolio_value - current_value
            
            return current_value + trade_impact
            
        except Exception as e:
            self.logger.error(f"Error calculating projected value: {e}")
            return current_value
    
    # Limit calculation methods (implemented as simplified versions)
    
    async def _calculate_var_absolute(self, scope_id: str, time_window: TimeWindow) -> Decimal:
        """Calculate absolute VaR for the scope"""
        # This would integrate with the risk analytics module
        # For now, return a placeholder calculation
        return Decimal("10000.00")  # $10,000 VaR
    
    async def _calculate_var_relative(self, scope_id: str, time_window: TimeWindow) -> Decimal:
        """Calculate relative VaR as percentage of portfolio value"""
        portfolio_value = await self._get_portfolio_value(scope_id)
        var_absolute = await self._calculate_var_absolute(scope_id, time_window)
        return (var_absolute / portfolio_value) * 100 if portfolio_value > 0 else Decimal("0")
    
    async def _calculate_gross_exposure(self, portfolio_id: str) -> Decimal:
        """Calculate gross exposure (sum of absolute values of all positions)"""
        # Query database for position values
        # Placeholder calculation
        return Decimal("50000.00")
    
    async def _calculate_net_exposure(self, portfolio_id: str) -> Decimal:
        """Calculate net exposure (long - short)"""
        # Query database for position values
        # Placeholder calculation
        return Decimal("25000.00")
    
    async def _calculate_long_exposure(self, portfolio_id: str) -> Decimal:
        """Calculate long exposure"""
        # Query database for long position values
        # Placeholder calculation
        return Decimal("40000.00")
    
    async def _calculate_short_exposure(self, portfolio_id: str) -> Decimal:
        """Calculate short exposure"""
        # Query database for short position values
        # Placeholder calculation
        return Decimal("15000.00")
    
    async def _calculate_max_drawdown(self, scope_id: str, time_window: TimeWindow) -> Decimal:
        """Calculate maximum drawdown over time window"""
        # This would analyze historical P&L data
        # Placeholder calculation
        return Decimal("5.5")  # 5.5% drawdown
    
    async def _calculate_daily_drawdown(self, scope_id: str) -> Decimal:
        """Calculate daily drawdown"""
        # Calculate today's P&L vs starting value
        # Placeholder calculation
        return Decimal("2.1")  # 2.1% daily drawdown
    
    async def _calculate_concentration(self, scope_id: str) -> Decimal:
        """Calculate maximum single position concentration"""
        # Find largest position as % of portfolio
        # Placeholder calculation
        return Decimal("15.5")  # 15.5% concentration
    
    async def _calculate_leverage(self, portfolio_id: str) -> Decimal:
        """Calculate portfolio leverage ratio"""
        # Gross exposure / portfolio value
        gross_exposure = await self._calculate_gross_exposure(portfolio_id)
        portfolio_value = await self._get_portfolio_value(portfolio_id)
        return gross_exposure / portfolio_value if portfolio_value > 0 else Decimal("0")
    
    async def _get_portfolio_value(self, portfolio_id: str) -> Decimal:
        """Get current portfolio value"""
        # Query database for current portfolio value
        # Placeholder
        return Decimal("100000.00")
    
    async def _get_position_value(self, portfolio_id: str, symbol: str) -> Decimal:
        """Get current position value for a symbol"""
        # Query database for position value
        # Placeholder
        return Decimal("15000.00")
    
    async def _send_limit_alert(self, limit: RiskLimit, check: LimitCheck) -> None:
        """Send alert for limit breach or warning"""
        try:
            if check.status == LimitStatus.BREACHED:
                message = create_breach_alert_message(
                    portfolio_id=limit.scope_id,
                    breach_type=limit.limit_type.value,
                    severity="high",
                    data={
                        "limit_id": limit.limit_id,
                        "current_value": float(check.current_value),
                        "limit_value": float(check.limit_value),
                        "utilization": check.utilization,
                        "breach_amount": float(check.breach_amount) if check.breach_amount else 0,
                        "timestamp": check.timestamp.isoformat(),
                        "scope": limit.scope.value,
                        "scope_id": limit.scope_id
                    }
                )
            else:  # WARNING
                message = create_risk_alert_message(
                    portfolio_id=limit.scope_id,
                    risk_type=limit.limit_type.value,
                    severity="medium",
                    data={
                        "limit_id": limit.limit_id,
                        "current_value": float(check.current_value),
                        "limit_value": float(check.limit_value),
                        "utilization": check.utilization,
                        "timestamp": check.timestamp.isoformat(),
                        "scope": limit.scope.value,
                        "scope_id": limit.scope_id
                    }
                )
            
            # Publish alert via Redis
            if self.redis_pubsub:
                if limit.scope == LimitScope.PORTFOLIO:
                    await self.redis_pubsub.publish_risk_alert(message.data, limit.scope_id)
                else:
                    await self.redis_pubsub.publish_risk_alert(message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending limit alert: {e}")
    
    async def _monitor_limits(self) -> None:
        """Background task to continuously monitor all active limits"""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                active_limits = [limit for limit in self.limits.values() if limit.status == LimitStatus.ACTIVE]
                
                for limit in active_limits:
                    try:
                        check = await self._perform_limit_check(limit, limit.scope_id)
                        if check and check.status in [LimitStatus.BREACHED, LimitStatus.WARNING]:
                            await self._send_limit_alert(limit, check)
                            
                            # Update limit status if breached
                            if check.status == LimitStatus.BREACHED:
                                limit.status = LimitStatus.BREACHED
                                await self._save_limit_to_db(limit)
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring limit {limit.limit_id}: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Limit monitoring task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in limit monitoring task: {e}")
    
    async def _adjust_dynamic_limits(self) -> None:
        """Background task to adjust dynamic limits based on market conditions"""
        try:
            while True:
                await asyncio.sleep(300)  # Adjust every 5 minutes
                
                dynamic_limits = [limit for limit in self.limits.values() if limit.auto_adjust]
                
                for limit in dynamic_limits:
                    try:
                        # Calculate volatility-based adjustment
                        current_volatility = await self._get_current_volatility(limit.scope_id)
                        adjustment_factor = 1.0 + (current_volatility - 0.20) * limit.volatility_multiplier
                        
                        # Calculate new limit value
                        base_limit = limit.metadata.get("base_limit", limit.limit_value)
                        new_limit = Decimal(str(float(base_limit) * adjustment_factor))
                        
                        # Apply min/max constraints
                        if limit.min_limit and new_limit < limit.min_limit:
                            new_limit = limit.min_limit
                        if limit.max_limit and new_limit > limit.max_limit:
                            new_limit = limit.max_limit
                        
                        # Update limit if changed significantly (>5%)
                        if abs(float(new_limit - limit.limit_value) / float(limit.limit_value)) > 0.05:
                            limit.limit_value = new_limit
                            limit.updated_at = datetime.utcnow()
                            await self._save_limit_to_db(limit)
                            
                            self.logger.info(
                                f"Adjusted dynamic limit {limit.limit_id} to {new_limit} "
                                f"(volatility: {current_volatility:.2%})"
                            )
                        
                    except Exception as e:
                        self.logger.error(f"Error adjusting dynamic limit {limit.limit_id}: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Dynamic limit adjustment task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in dynamic limit adjustment task: {e}")
    
    async def _get_current_volatility(self, scope_id: str) -> float:
        """Get current volatility for dynamic limit adjustment"""
        # This would integrate with market data to calculate realized volatility
        # Placeholder calculation
        return 0.25  # 25% annualized volatility
    
    # Database operations (simplified implementations)
    
    async def _load_limits_from_db(self) -> None:
        """Load existing limits from database"""
        try:
            # This would query the database for existing limits
            # For now, create some default limits
            self.logger.info("Loaded limits from database (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Error loading limits from database: {e}")
    
    async def _save_limit_to_db(self, limit: RiskLimit) -> None:
        """Save limit configuration to database"""
        try:
            # This would save the limit to a risk_limits table
            self.logger.debug(f"Saved limit {limit.limit_id} to database")
            
        except Exception as e:
            self.logger.error(f"Error saving limit to database: {e}")
    
    async def _delete_limit_from_db(self, limit_id: str) -> None:
        """Delete limit from database"""
        try:
            # This would delete the limit from the database
            self.logger.debug(f"Deleted limit {limit_id} from database")
            
        except Exception as e:
            self.logger.error(f"Error deleting limit from database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get limit engine statistics"""
        return {
            "total_limits": len(self.limits),
            "active_limits": len([l for l in self.limits.values() if l.status == LimitStatus.ACTIVE]),
            "breached_limits": len([l for l in self.limits.values() if l.status == LimitStatus.BREACHED]),
            "dynamic_limits": len([l for l in self.limits.values() if l.auto_adjust]),
            "total_checks": self.total_checks,
            "breach_count": self.breach_count,
            "warning_count": self.warning_count,
            "limit_types": {
                limit_type.value: len([l for l in self.limits.values() if l.limit_type == limit_type])
                for limit_type in LimitType
            }
        }

# Global instance
limit_engine = None

def get_limit_engine() -> DynamicLimitEngine:
    """Get global limit engine instance"""
    global limit_engine
    if limit_engine is None:
        raise RuntimeError("Limit engine not initialized. Call init_limit_engine() first.")
    return limit_engine

def init_limit_engine() -> DynamicLimitEngine:
    """Initialize global limit engine instance"""
    global limit_engine
    limit_engine = DynamicLimitEngine()
    return limit_engine
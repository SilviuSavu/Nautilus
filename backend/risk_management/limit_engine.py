"""
Dynamic Risk Limit Enforcement Engine
Configurable risk limits with real-time checking and automatic actions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, asdict
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Types of risk limits"""
    VAR = "var"
    CONCENTRATION = "concentration"
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    CORRELATION = "correlation"
    EXPOSURE = "exposure"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    NOTIONAL = "notional"

class LimitAction(Enum):
    """Actions to take when limits are breached"""
    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"
    NOTIFY = "notify"
    LIQUIDATE = "liquidate"
    FREEZE = "freeze"

class LimitSeverity(Enum):
    """Severity levels for limit breaches"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    id: str
    name: str
    portfolio_id: Optional[str]  # None for global limits
    user_id: Optional[str]  # None for portfolio-level limits
    strategy_id: Optional[str]  # None for user/portfolio limits
    limit_type: LimitType
    threshold_value: Decimal
    warning_threshold: Decimal
    action: LimitAction
    active: bool
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    # Time-based constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Additional parameters for specific limit types
    parameters: Dict[str, Any] = None
    
    # Breach tracking
    breach_count: int = 0
    last_breach: Optional[datetime] = None
    escalation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert enums and special types
        result['limit_type'] = self.limit_type.value
        result['action'] = self.action.value
        result['threshold_value'] = str(self.threshold_value)
        result['warning_threshold'] = str(self.warning_threshold)
        
        for field in ['created_at', 'updated_at', 'start_time', 'end_time', 'last_breach']:
            if result[field]:
                result[field] = result[field].isoformat()
        
        return result

@dataclass
class LimitBreach:
    """Record of a limit breach"""
    id: str
    limit_id: str
    portfolio_id: str
    breach_type: str  # 'warning' or 'critical'
    current_value: Decimal
    threshold_value: Decimal
    severity: LimitSeverity
    timestamp: datetime
    action_taken: LimitAction
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['current_value'] = str(self.current_value)
        result['threshold_value'] = str(self.threshold_value)
        result['severity'] = self.severity.value
        result['action_taken'] = self.action_taken.value
        result['timestamp'] = self.timestamp.isoformat()
        if result['resolved_at']:
            result['resolved_at'] = result['resolved_at'].isoformat()
        return result

class LimitEngine:
    """
    Dynamic risk limit enforcement engine with real-time checking
    and configurable actions
    """
    
    def __init__(self, risk_monitor=None, position_service=None, order_service=None):
        self.risk_monitor = risk_monitor
        self.position_service = position_service
        self.order_service = order_service
        
        # Limit storage
        self.limits: Dict[str, RiskLimit] = {}
        self.breaches: Dict[str, LimitBreach] = {}
        
        # Active monitoring
        self.monitoring_active = False
        self._monitoring_task = None
        
        # Escalation callbacks
        self._escalation_callbacks: List[Callable] = []
        
        # Breach history for analysis
        self.breach_history: List[LimitBreach] = []
        
        # Performance tracking
        self.check_count = 0
        self.breach_prevention_count = 0
        
    async def add_limit(self, limit: RiskLimit) -> str:
        """Add a new risk limit"""
        try:
            # Validate limit configuration
            await self._validate_limit(limit)
            
            # Store limit
            self.limits[limit.id] = limit
            
            logger.info(f"Added risk limit: {limit.name} ({limit.id})")
            return limit.id
            
        except Exception as e:
            logger.error(f"Error adding risk limit: {e}")
            raise
    
    async def update_limit(self, limit_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing risk limit"""
        try:
            if limit_id not in self.limits:
                raise ValueError(f"Limit {limit_id} not found")
            
            limit = self.limits[limit_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(limit, field):
                    setattr(limit, field, value)
            
            limit.updated_at = datetime.utcnow()
            
            logger.info(f"Updated risk limit: {limit_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk limit: {e}")
            raise
    
    async def remove_limit(self, limit_id: str) -> bool:
        """Remove a risk limit"""
        try:
            if limit_id in self.limits:
                del self.limits[limit_id]
                logger.info(f"Removed risk limit: {limit_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing risk limit: {e}")
            raise
    
    async def start_monitoring(self):
        """Start real-time limit monitoring"""
        try:
            logger.info("Starting limit engine monitoring")
            
            self.monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Limit engine monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting limit monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop real-time limit monitoring"""
        try:
            logger.info("Stopping limit engine monitoring")
            
            self.monitoring_active = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Limit engine monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping limit monitoring: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop for limit checking"""
        try:
            while self.monitoring_active:
                start_time = datetime.utcnow()
                
                # Check all active limits
                await self._check_all_limits()
                
                # Clean up old breaches
                await self._cleanup_old_breaches()
                
                # Sleep for 1 second interval
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                sleep_time = max(0, 1.0 - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Limit monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in limit monitoring loop: {e}")
            self.monitoring_active = False
    
    async def _check_all_limits(self):
        """Check all active limits for breaches"""
        try:
            for limit in self.limits.values():
                if not limit.active:
                    continue
                
                # Check time constraints
                now = datetime.utcnow()
                if limit.start_time and now < limit.start_time:
                    continue
                if limit.end_time and now > limit.end_time:
                    continue
                
                # Check the limit
                await self._check_limit(limit)
                
            self.check_count += 1
            
        except Exception as e:
            logger.error(f"Error checking all limits: {e}")
    
    async def _check_limit(self, limit: RiskLimit):
        """Check a specific limit for breaches"""
        try:
            # Get current value for the limit
            current_value = await self._get_current_value(limit)
            
            if current_value is None:
                return
            
            # Check for warning threshold breach
            if current_value >= limit.warning_threshold:
                await self._handle_warning_breach(limit, current_value)
            
            # Check for critical threshold breach
            if current_value >= limit.threshold_value:
                await self._handle_critical_breach(limit, current_value)
                
        except Exception as e:
            logger.error(f"Error checking limit {limit.id}: {e}")
    
    async def _get_current_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current value for a limit type"""
        try:
            if limit.limit_type == LimitType.VAR:
                return await self._get_var_value(limit)
            elif limit.limit_type == LimitType.CONCENTRATION:
                return await self._get_concentration_value(limit)
            elif limit.limit_type == LimitType.POSITION_SIZE:
                return await self._get_position_size_value(limit)
            elif limit.limit_type == LimitType.LEVERAGE:
                return await self._get_leverage_value(limit)
            elif limit.limit_type == LimitType.EXPOSURE:
                return await self._get_exposure_value(limit)
            elif limit.limit_type == LimitType.DRAWDOWN:
                return await self._get_drawdown_value(limit)
            elif limit.limit_type == LimitType.VOLATILITY:
                return await self._get_volatility_value(limit)
            elif limit.limit_type == LimitType.NOTIONAL:
                return await self._get_notional_value(limit)
            else:
                logger.warning(f"Unknown limit type: {limit.limit_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current value for limit {limit.id}: {e}")
            return None
    
    async def _get_var_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current VaR value"""
        try:
            if self.risk_monitor and limit.portfolio_id:
                snapshot = await self.risk_monitor.get_current_risk_metrics(limit.portfolio_id)
                if snapshot:
                    confidence = limit.parameters.get('confidence', 95) if limit.parameters else 95
                    if confidence == 95:
                        return snapshot.var_1d_95
                    elif confidence == 99:
                        return snapshot.var_1d_99
            return None
        except Exception as e:
            logger.error(f"Error getting VaR value: {e}")
            return None
    
    async def _get_concentration_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current concentration value"""
        try:
            if self.risk_monitor and limit.portfolio_id:
                snapshot = await self.risk_monitor.get_current_risk_metrics(limit.portfolio_id)
                if snapshot:
                    return Decimal(str(snapshot.max_position_concentration))
            return None
        except Exception as e:
            logger.error(f"Error getting concentration value: {e}")
            return None
    
    async def _get_position_size_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current position size value"""
        try:
            if self.position_service and limit.portfolio_id:
                positions = await self.position_service.get_positions(limit.portfolio_id)
                symbol = limit.parameters.get('symbol') if limit.parameters else None
                
                if symbol:
                    for pos in positions:
                        if pos.get('symbol') == symbol:
                            return abs(pos.get('market_value', Decimal('0')))
                else:
                    # Return largest position
                    if positions:
                        return max(abs(pos.get('market_value', Decimal('0'))) for pos in positions)
            return None
        except Exception as e:
            logger.error(f"Error getting position size value: {e}")
            return None
    
    async def _get_leverage_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current leverage value"""
        try:
            if self.position_service and limit.portfolio_id:
                positions = await self.position_service.get_positions(limit.portfolio_id)
                if positions:
                    total_exposure = sum(abs(pos.get('market_value', Decimal('0'))) for pos in positions)
                    # Mock equity value - in production would get from account service
                    equity = Decimal('100000')  # Mock 100k equity
                    if equity > 0:
                        return total_exposure / equity
            return None
        except Exception as e:
            logger.error(f"Error getting leverage value: {e}")
            return None
    
    async def _get_exposure_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current exposure value"""
        try:
            if self.risk_monitor and limit.portfolio_id:
                snapshot = await self.risk_monitor.get_current_risk_metrics(limit.portfolio_id)
                if snapshot:
                    exposure_type = limit.parameters.get('type', 'total') if limit.parameters else 'total'
                    if exposure_type == 'total':
                        return snapshot.total_exposure
                    elif exposure_type == 'net':
                        return abs(snapshot.net_exposure)
            return None
        except Exception as e:
            logger.error(f"Error getting exposure value: {e}")
            return None
    
    async def _get_drawdown_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current drawdown value"""
        # Mock implementation - would calculate from PnL history
        return Decimal('0.05')  # 5% drawdown
    
    async def _get_volatility_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current volatility value"""
        try:
            if self.risk_monitor and limit.portfolio_id:
                snapshot = await self.risk_monitor.get_current_risk_metrics(limit.portfolio_id)
                if snapshot:
                    return Decimal(str(snapshot.portfolio_volatility))
            return None
        except Exception as e:
            logger.error(f"Error getting volatility value: {e}")
            return None
    
    async def _get_notional_value(self, limit: RiskLimit) -> Optional[Decimal]:
        """Get current notional value"""
        try:
            if self.position_service and limit.portfolio_id:
                positions = await self.position_service.get_positions(limit.portfolio_id)
                if positions:
                    return sum(abs(pos.get('market_value', Decimal('0'))) for pos in positions)
            return None
        except Exception as e:
            logger.error(f"Error getting notional value: {e}")
            return None
    
    async def _handle_warning_breach(self, limit: RiskLimit, current_value: Decimal):
        """Handle warning threshold breach"""
        try:
            breach_id = f"warning_{limit.id}_{int(datetime.utcnow().timestamp())}"
            
            breach = LimitBreach(
                id=breach_id,
                limit_id=limit.id,
                portfolio_id=limit.portfolio_id or 'global',
                breach_type='warning',
                current_value=current_value,
                threshold_value=limit.warning_threshold,
                severity=LimitSeverity.WARNING,
                timestamp=datetime.utcnow(),
                action_taken=LimitAction.WARN
            )
            
            self.breaches[breach_id] = breach
            self.breach_history.append(breach)
            
            logger.warning(f"Warning breach for limit {limit.name}: {current_value} >= {limit.warning_threshold}")
            
        except Exception as e:
            logger.error(f"Error handling warning breach: {e}")
    
    async def _handle_critical_breach(self, limit: RiskLimit, current_value: Decimal):
        """Handle critical threshold breach"""
        try:
            breach_id = f"critical_{limit.id}_{int(datetime.utcnow().timestamp())}"
            
            breach = LimitBreach(
                id=breach_id,
                limit_id=limit.id,
                portfolio_id=limit.portfolio_id or 'global',
                breach_type='critical',
                current_value=current_value,
                threshold_value=limit.threshold_value,
                severity=LimitSeverity.CRITICAL,
                timestamp=datetime.utcnow(),
                action_taken=limit.action
            )
            
            self.breaches[breach_id] = breach
            self.breach_history.append(breach)
            
            # Update limit breach tracking
            limit.breach_count += 1
            limit.last_breach = datetime.utcnow()
            
            # Execute the configured action
            await self._execute_limit_action(limit, breach, current_value)
            
            logger.critical(f"Critical breach for limit {limit.name}: {current_value} >= {limit.threshold_value}")
            
        except Exception as e:
            logger.error(f"Error handling critical breach: {e}")
    
    async def _execute_limit_action(self, limit: RiskLimit, breach: LimitBreach, current_value: Decimal):
        """Execute the configured action for a limit breach"""
        try:
            if limit.action == LimitAction.WARN:
                # Already logged the warning
                pass
            elif limit.action == LimitAction.BLOCK:
                await self._block_new_orders(limit.portfolio_id)
            elif limit.action == LimitAction.REDUCE:
                await self._reduce_positions(limit, current_value)
            elif limit.action == LimitAction.NOTIFY:
                await self._send_notifications(limit, breach)
            elif limit.action == LimitAction.LIQUIDATE:
                await self._liquidate_positions(limit)
            elif limit.action == LimitAction.FREEZE:
                await self._freeze_portfolio(limit.portfolio_id)
            
            self.breach_prevention_count += 1
            
        except Exception as e:
            logger.error(f"Error executing limit action: {e}")
    
    async def _block_new_orders(self, portfolio_id: str):
        """Block new orders for a portfolio"""
        logger.warning(f"Blocking new orders for portfolio {portfolio_id}")
        # Implementation would integrate with order management system
    
    async def _reduce_positions(self, limit: RiskLimit, current_value: Decimal):
        """Reduce positions to bring within limits"""
        logger.warning(f"Reducing positions for limit {limit.name}")
        # Implementation would calculate reduction needed and execute trades
    
    async def _send_notifications(self, limit: RiskLimit, breach: LimitBreach):
        """Send notifications for limit breach"""
        logger.warning(f"Sending notifications for limit breach: {limit.name}")
        # Implementation would send emails/SMS/webhooks
    
    async def _liquidate_positions(self, limit: RiskLimit):
        """Liquidate positions for emergency limit breach"""
        logger.critical(f"Liquidating positions for limit {limit.name}")
        # Implementation would execute emergency liquidation
    
    async def _freeze_portfolio(self, portfolio_id: str):
        """Freeze all trading activity for a portfolio"""
        logger.critical(f"Freezing portfolio {portfolio_id}")
        # Implementation would disable all trading
    
    async def _cleanup_old_breaches(self):
        """Clean up old breach records"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Remove old unresolved breaches
            old_breaches = [
                breach_id for breach_id, breach in self.breaches.items()
                if breach.timestamp < cutoff_time and not breach.resolved
            ]
            
            for breach_id in old_breaches:
                del self.breaches[breach_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old breaches: {e}")
    
    async def _validate_limit(self, limit: RiskLimit):
        """Validate limit configuration"""
        if limit.warning_threshold >= limit.threshold_value:
            raise ValueError("Warning threshold must be less than critical threshold")
        
        if limit.threshold_value <= 0:
            raise ValueError("Threshold value must be positive")
        
        # Type-specific validations
        if limit.limit_type == LimitType.CONCENTRATION:
            if limit.threshold_value > 1:
                raise ValueError("Concentration limit cannot exceed 100%")
    
    async def check_pre_trade_limits(self, portfolio_id: str, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a proposed trade would breach any limits"""
        try:
            violations = []
            
            # Get applicable limits
            applicable_limits = [
                limit for limit in self.limits.values()
                if limit.active and (limit.portfolio_id == portfolio_id or limit.portfolio_id is None)
            ]
            
            # Simulate trade impact and check each limit
            for limit in applicable_limits:
                projected_value = await self._project_limit_value_after_trade(limit, trade_request)
                
                if projected_value and projected_value >= limit.threshold_value:
                    violations.append({
                        'limit_id': limit.id,
                        'limit_name': limit.name,
                        'limit_type': limit.limit_type.value,
                        'current_value': str(await self._get_current_value(limit) or 0),
                        'projected_value': str(projected_value),
                        'threshold': str(limit.threshold_value),
                        'action': limit.action.value
                    })
            
            return {
                'violations': violations,
                'trade_approved': len(violations) == 0,
                'blocking_violations': [v for v in violations if v['action'] in ['block', 'liquidate', 'freeze']]
            }
            
        except Exception as e:
            logger.error(f"Error checking pre-trade limits: {e}")
            return {'violations': [], 'trade_approved': False, 'error': str(e)}
    
    async def _project_limit_value_after_trade(self, limit: RiskLimit, trade_request: Dict[str, Any]) -> Optional[Decimal]:
        """Project what the limit value would be after a trade"""
        # Simplified projection - would be more sophisticated in production
        current_value = await self._get_current_value(limit)
        
        if current_value is None:
            return None
        
        # Mock impact calculation
        trade_size = abs(Decimal(str(trade_request.get('quantity', 0))))
        price = Decimal(str(trade_request.get('price', 100)))
        trade_value = trade_size * price
        
        if limit.limit_type in [LimitType.EXPOSURE, LimitType.NOTIONAL]:
            return current_value + trade_value
        elif limit.limit_type == LimitType.POSITION_SIZE:
            symbol = trade_request.get('symbol')
            if limit.parameters and limit.parameters.get('symbol') == symbol:
                return current_value + trade_value
        
        return current_value
    
    async def get_limit_status(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of all limits"""
        try:
            applicable_limits = []
            
            for limit in self.limits.values():
                if portfolio_id is None or limit.portfolio_id == portfolio_id or limit.portfolio_id is None:
                    current_value = await self._get_current_value(limit)
                    
                    limit_status = {
                        **limit.to_dict(),
                        'current_value': str(current_value) if current_value else None,
                        'utilization_pct': float(current_value / limit.threshold_value * 100) if current_value and limit.threshold_value else 0,
                        'warning_utilization_pct': float(current_value / limit.warning_threshold * 100) if current_value and limit.warning_threshold else 0
                    }
                    
                    applicable_limits.append(limit_status)
            
            active_breaches = [breach.to_dict() for breach in self.breaches.values() if not breach.resolved]
            
            return {
                'limits': applicable_limits,
                'active_breaches': active_breaches,
                'monitoring_active': self.monitoring_active,
                'total_limits': len(applicable_limits),
                'total_breaches': len(active_breaches),
                'check_count': self.check_count,
                'breach_prevention_count': self.breach_prevention_count
            }
            
        except Exception as e:
            logger.error(f"Error getting limit status: {e}")
            return {'error': str(e)}
    
    def add_escalation_callback(self, callback: Callable):
        """Add callback for limit escalations"""
        self._escalation_callbacks.append(callback)

# Global instance
limit_engine = LimitEngine()
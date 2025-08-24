"""
Real-Time Margin Monitor
=======================

High-performance margin monitoring system with:
- Real-time margin calculations with M4 Max acceleration
- Predictive margin call alerts
- Integration with existing MessageBus infrastructure
- Advanced alerting with multiple severity levels
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from models import (
    Portfolio, Position, MarginAlert, AlertSeverity, RecommendedAction,
    MarginRequirement
)
from margin_calculator import MarginCalculator
from collateral_optimizer import CollateralOptimizer

# Import M4 Max hardware acceleration
try:
    from backend.hardware_router import hardware_accelerated, WorkloadType
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False
    
    # Create dummy WorkloadType enum
    class WorkloadType:
        RISK_CALCULATION = "risk_calculation"
        MONTE_CARLO = "monte_carlo"
        ML_INFERENCE = "ml_inference"
    
    def hardware_accelerated(workload_type, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import MessageBus if available
try:
    from backend.messagebus_client import get_messagebus_client
    MESSAGEBUS_AVAILABLE = True
except ImportError:
    MESSAGEBUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for margin monitoring"""
    update_interval_seconds: int = 5  # Monitor every 5 seconds
    warning_threshold: Decimal = Decimal('0.80')  # 80% margin utilization warning
    critical_threshold: Decimal = Decimal('0.90')  # 90% margin utilization critical
    emergency_threshold: Decimal = Decimal('0.95')  # 95% margin utilization emergency
    predictive_horizon_minutes: int = 60  # Predict margin calls 60 minutes ahead
    alert_cooldown_minutes: int = 5  # Minimum time between similar alerts
    enable_predictive_alerts: bool = True
    enable_stress_testing: bool = True


class RealTimeMarginMonitor:
    """
    Real-time margin monitoring system with M4 Max hardware acceleration.
    
    Features:
    - Continuous monitoring of margin levels
    - Predictive margin call analysis
    - Multiple alert severity levels
    - Integration with MessageBus for real-time notifications
    - Hardware-accelerated calculations for sub-second response times
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.margin_calculator = MarginCalculator()
        self.collateral_optimizer = CollateralOptimizer()
        
        # Monitoring state
        self.monitoring_tasks = {}
        self.alert_history = {}
        self.last_calculations = {}
        self.alert_callbacks = []
        
        # MessageBus integration
        self.messagebus_client = None
        if MESSAGEBUS_AVAILABLE:
            try:
                self.messagebus_client = get_messagebus_client()
            except Exception as e:
                logger.warning(f"Failed to initialize MessageBus client: {e}")
    
    async def start_monitoring(self, portfolio: Portfolio, alert_callback: Optional[Callable] = None):
        """Start real-time monitoring for a portfolio"""
        
        if portfolio.id in self.monitoring_tasks:
            logger.warning(f"Monitoring already active for portfolio {portfolio.id}")
            return
        
        if alert_callback:
            self.alert_callbacks.append(alert_callback)
        
        # Start monitoring task
        task = asyncio.create_task(self._monitor_portfolio_loop(portfolio))
        self.monitoring_tasks[portfolio.id] = task
        
        logger.info(f"Started real-time margin monitoring for portfolio {portfolio.id}")
    
    async def stop_monitoring(self, portfolio_id: str):
        """Stop monitoring for a portfolio"""
        if portfolio_id in self.monitoring_tasks:
            task = self.monitoring_tasks[portfolio_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.monitoring_tasks[portfolio_id]
            logger.info(f"Stopped margin monitoring for portfolio {portfolio_id}")
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=5000)
    async def _monitor_portfolio_loop(self, portfolio: Portfolio):
        """Main monitoring loop for a portfolio"""
        
        while True:
            try:
                start_time = time.time()
                
                # Calculate current margin requirements
                margin_req = await self.margin_calculator.calculate_portfolio_margin_requirement(portfolio)
                
                # Check for alerts
                alerts = await self._analyze_margin_alerts(portfolio, margin_req)
                
                # Process any alerts
                for alert in alerts:
                    await self._process_alert(alert)
                
                # Store latest calculation
                self.last_calculations[portfolio.id] = {
                    'margin_requirement': margin_req,
                    'calculated_at': datetime.now(timezone.utc),
                    'calculation_time_ms': (time.time() - start_time) * 1000
                }
                
                # Publish to MessageBus if available
                if self.messagebus_client:
                    await self._publish_margin_update(portfolio.id, margin_req, alerts)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"Monitoring cancelled for portfolio {portfolio.id}")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop for portfolio {portfolio.id}: {e}")
                await asyncio.sleep(self.config.update_interval_seconds)
    
    async def _analyze_margin_alerts(self, portfolio: Portfolio, margin_req: MarginRequirement) -> List[MarginAlert]:
        """Analyze margin requirements and generate alerts"""
        alerts = []
        
        # Check current margin utilization
        current_alert = await self._check_current_margin_levels(portfolio, margin_req)
        if current_alert:
            alerts.append(current_alert)
        
        # Predictive alerts
        if self.config.enable_predictive_alerts:
            predictive_alerts = await self._check_predictive_margin_calls(portfolio, margin_req)
            alerts.extend(predictive_alerts)
        
        # Stress test alerts
        if self.config.enable_stress_testing:
            stress_alert = await self._check_stress_test_margins(portfolio)
            if stress_alert:
                alerts.append(stress_alert)
        
        return alerts
    
    async def _check_current_margin_levels(self, portfolio: Portfolio, margin_req: MarginRequirement) -> Optional[MarginAlert]:
        """Check current margin levels against thresholds"""
        
        utilization = margin_req.margin_utilization
        
        # Determine severity based on utilization
        if utilization >= self.config.emergency_threshold:
            severity = AlertSeverity.EMERGENCY
            message = f"EMERGENCY: Margin utilization at {margin_req.margin_utilization_percent:.1f}% - Immediate liquidation risk"
            recommended_action = RecommendedAction.LIQUIDATE_POSITIONS
        elif utilization >= self.config.critical_threshold:
            severity = AlertSeverity.CRITICAL
            message = f"CRITICAL: Margin utilization at {margin_req.margin_utilization_percent:.1f}% - Margin call imminent"
            recommended_action = RecommendedAction.IMMEDIATE_ACTION_REQUIRED
        elif utilization >= self.config.warning_threshold:
            severity = AlertSeverity.WARNING
            message = f"WARNING: High margin utilization at {margin_req.margin_utilization_percent:.1f}%"
            recommended_action = RecommendedAction.REDUCE_POSITIONS
        else:
            return None  # No alert needed
        
        # Check alert cooldown
        if self._is_alert_in_cooldown(portfolio.id, severity):
            return None
        
        # Calculate required action amount
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            # Calculate amount needed to reduce to warning level
            target_utilization = self.config.warning_threshold
            required_margin_reduction = margin_req.total_margin_requirement * (utilization - target_utilization)
            required_action_amount = required_margin_reduction
        else:
            required_action_amount = None
        
        alert = MarginAlert(
            portfolio_id=portfolio.id,
            severity=severity,
            message=message,
            margin_utilization=utilization,
            time_to_margin_call_minutes=margin_req.time_to_margin_call_minutes,
            recommended_action=recommended_action,
            affected_positions=self._get_high_risk_positions(portfolio),
            required_action_amount=required_action_amount
        )
        
        return alert
    
    @hardware_accelerated(WorkloadType.ML_INFERENCE, data_size=2000)
    async def _check_predictive_margin_calls(self, portfolio: Portfolio, margin_req: MarginRequirement) -> List[MarginAlert]:
        """Use predictive analytics to forecast margin calls"""
        alerts = []
        
        # Get historical volatility and trends
        portfolio_volatility = await self._calculate_portfolio_volatility(portfolio)
        
        # Predict margin utilization over time
        predictions = await self._predict_margin_trajectory(portfolio, margin_req, portfolio_volatility)
        
        for prediction in predictions:
            if prediction['predicted_utilization'] >= self.config.critical_threshold:
                time_to_critical = prediction['time_minutes']
                
                if time_to_critical <= self.config.predictive_horizon_minutes:
                    alert = MarginAlert(
                        portfolio_id=portfolio.id,
                        severity=AlertSeverity.WARNING,
                        message=f"PREDICTIVE: Margin call predicted in {time_to_critical} minutes "
                               f"(utilization will reach {prediction['predicted_utilization']*100:.1f}%)",
                        margin_utilization=margin_req.margin_utilization,
                        time_to_margin_call_minutes=time_to_critical,
                        recommended_action=RecommendedAction.MONITOR_CLOSELY,
                        affected_positions=self._get_volatile_positions(portfolio)
                    )
                    alerts.append(alert)
                    break  # Only send one predictive alert
        
        return alerts
    
    async def _calculate_portfolio_volatility(self, portfolio: Portfolio) -> Decimal:
        """Calculate portfolio volatility for predictive analysis"""
        # Mock calculation - in production, use historical returns
        total_volatility = Decimal('0')
        total_weight = Decimal('0')
        
        for position in portfolio.positions:
            # Mock volatility based on asset class
            asset_vol = self._get_mock_asset_volatility(position)
            weight = position.notional_value / portfolio.gross_exposure
            
            total_volatility += asset_vol * weight
            total_weight += weight
        
        return total_volatility / total_weight if total_weight > 0 else Decimal('0.02')
    
    def _get_mock_asset_volatility(self, position: Position) -> Decimal:
        """Mock asset volatility - replace with real market data"""
        volatilities = {
            'EQUITY': Decimal('0.20'),     # 20% annual volatility
            'BOND': Decimal('0.05'),       # 5% annual volatility
            'FX': Decimal('0.10'),         # 10% annual volatility
            'COMMODITY': Decimal('0.25'),  # 25% annual volatility
            'DERIVATIVE': Decimal('0.30'), # 30% annual volatility
            'CRYPTO': Decimal('0.60')      # 60% annual volatility
        }
        
        return volatilities.get(position.asset_class.value.upper(), Decimal('0.15'))
    
    async def _predict_margin_trajectory(self, portfolio: Portfolio, margin_req: MarginRequirement, volatility: Decimal) -> List[Dict]:
        """Predict margin utilization trajectory"""
        predictions = []
        current_utilization = margin_req.margin_utilization
        
        # Predict for next hour in 10-minute intervals
        for minutes in range(10, 61, 10):
            # Monte Carlo-style prediction with volatility
            time_factor = Decimal(str(minutes)) / Decimal('525600')  # Convert to years
            volatility_adjustment = volatility * (time_factor.sqrt() if time_factor > 0 else Decimal('0'))
            
            # Assume worst-case scenario (2 standard deviations)
            predicted_utilization = current_utilization + (volatility_adjustment * 2)
            
            predictions.append({
                'time_minutes': minutes,
                'predicted_utilization': predicted_utilization,
                'confidence': Decimal('0.95')  # 95% confidence interval
            })
        
        return predictions
    
    async def _check_stress_test_margins(self, portfolio: Portfolio) -> Optional[MarginAlert]:
        """Check margin requirements under stress scenarios"""
        try:
            stress_result = await self.margin_calculator.run_margin_stress_test(portfolio, "Daily Stress Test")
            
            if not stress_result.passes_stress_test:
                return MarginAlert(
                    portfolio_id=portfolio.id,
                    severity=AlertSeverity.INFO,
                    message=f"STRESS TEST: Portfolio fails stress test with {stress_result.margin_increase_percent*100:.0f}% margin increase",
                    margin_utilization=Decimal('0'),  # Stress test, not current utilization
                    recommended_action=RecommendedAction.MONITOR_CLOSELY,
                    affected_positions=stress_result.positions_at_risk
                )
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
        
        return None
    
    def _get_high_risk_positions(self, portfolio: Portfolio) -> List[str]:
        """Identify positions contributing most to margin requirement"""
        # Mock implementation - in production, calculate actual margin contributions
        high_risk = []
        for position in portfolio.positions:
            if position.asset_class.value in ['CRYPTO', 'DERIVATIVE'] or position.notional_value > Decimal('1000000'):
                high_risk.append(position.id)
        return high_risk[:5]  # Return top 5 high-risk positions
    
    def _get_volatile_positions(self, portfolio: Portfolio) -> List[str]:
        """Identify most volatile positions"""
        volatile = []
        for position in portfolio.positions:
            volatility = self._get_mock_asset_volatility(position)
            if volatility > Decimal('0.25'):  # 25% volatility threshold
                volatile.append(position.id)
        return volatile
    
    def _is_alert_in_cooldown(self, portfolio_id: str, severity: AlertSeverity) -> bool:
        """Check if alert is in cooldown period"""
        alert_key = f"{portfolio_id}_{severity.value}"
        
        if alert_key in self.alert_history:
            last_alert_time = self.alert_history[alert_key]
            cooldown_period = timedelta(minutes=self.config.alert_cooldown_minutes)
            
            if datetime.now(timezone.utc) - last_alert_time < cooldown_period:
                return True
        
        return False
    
    async def _process_alert(self, alert: MarginAlert):
        """Process and distribute alerts"""
        
        # Update alert history
        alert_key = f"{alert.portfolio_id}_{alert.severity.value}"
        self.alert_history[alert_key] = alert.created_at
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        logger.log(log_level, f"Margin Alert: {alert.message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    async def _publish_margin_update(self, portfolio_id: str, margin_req: MarginRequirement, alerts: List[MarginAlert]):
        """Publish margin update to MessageBus"""
        if not self.messagebus_client:
            return
        
        try:
            message = {
                'type': 'margin_update',
                'portfolio_id': portfolio_id,
                'margin_requirement': {
                    'total_margin': float(margin_req.total_margin_requirement),
                    'margin_utilization': float(margin_req.margin_utilization),
                    'margin_excess': float(margin_req.margin_excess),
                    'time_to_margin_call_minutes': margin_req.time_to_margin_call_minutes
                },
                'alerts': [
                    {
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'recommended_action': alert.recommended_action.value if alert.recommended_action else None
                    }
                    for alert in alerts
                ],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.messagebus_client.publish('margin_monitoring', message)
            
        except Exception as e:
            logger.error(f"Error publishing margin update to MessageBus: {e}")
    
    async def get_monitoring_status(self, portfolio_id: str) -> Dict[str, Any]:
        """Get current monitoring status for a portfolio"""
        
        is_monitoring = portfolio_id in self.monitoring_tasks
        last_calculation = self.last_calculations.get(portfolio_id)
        
        status = {
            'is_monitoring': is_monitoring,
            'last_update': last_calculation['calculated_at'].isoformat() if last_calculation else None,
            'calculation_time_ms': last_calculation['calculation_time_ms'] if last_calculation else None,
            'active_alerts': len([alert for alert in self.alert_history.keys() if alert.startswith(portfolio_id)]),
            'config': {
                'update_interval_seconds': self.config.update_interval_seconds,
                'warning_threshold': float(self.config.warning_threshold),
                'critical_threshold': float(self.config.critical_threshold),
                'emergency_threshold': float(self.config.emergency_threshold)
            }
        }
        
        if last_calculation:
            margin_req = last_calculation['margin_requirement']
            status['current_margin'] = {
                'total_margin_requirement': float(margin_req.total_margin_requirement),
                'margin_utilization': float(margin_req.margin_utilization),
                'margin_utilization_percent': margin_req.margin_utilization_percent,
                'margin_excess': float(margin_req.margin_excess),
                'time_to_margin_call_minutes': margin_req.time_to_margin_call_minutes
            }
        
        return status
    
    async def emergency_stop_monitoring(self):
        """Emergency stop all monitoring (for system shutdown)"""
        logger.warning("Emergency stop: Cancelling all margin monitoring tasks")
        
        tasks = list(self.monitoring_tasks.values())
        self.monitoring_tasks.clear()
        
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All margin monitoring tasks stopped")


# Global margin monitor instance
_global_margin_monitor = None


def get_margin_monitor() -> RealTimeMarginMonitor:
    """Get the global margin monitor instance"""
    global _global_margin_monitor
    if _global_margin_monitor is None:
        _global_margin_monitor = RealTimeMarginMonitor()
    return _global_margin_monitor
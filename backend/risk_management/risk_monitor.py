"""
Real-time Risk Monitoring Service
Continuous position monitoring with real-time exposure calculations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from decimal import Decimal
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RiskMetricSnapshot:
    """Snapshot of risk metrics at a point in time"""
    timestamp: datetime
    portfolio_id: str
    total_exposure: Decimal
    net_exposure: Decimal
    var_1d_95: Decimal
    var_1d_99: Decimal
    portfolio_beta: float
    portfolio_volatility: float
    max_position_concentration: float
    correlation_risk_score: float
    liquidity_risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert Decimal to string for JSON serialization
        for key, value in result.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
        return result

@dataclass
class PositionUpdate:
    """Real-time position update notification"""
    portfolio_id: str
    symbol: str
    quantity: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    timestamp: datetime
    update_type: str  # 'new', 'modified', 'closed'

class RiskMonitor:
    """
    Real-time risk monitoring service with continuous position tracking
    and risk metric updates
    """
    
    def __init__(self, websocket_manager=None, position_service=None):
        self.websocket_manager = websocket_manager
        self.position_service = position_service
        
        # Risk metric snapshots by portfolio
        self.risk_snapshots: Dict[str, List[RiskMetricSnapshot]] = defaultdict(list)
        
        # Current positions by portfolio
        self.current_positions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Risk calculation frequency
        self.update_interval = 1.0  # seconds
        self.monitoring_active = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'var_increase_pct': 20.0,  # Alert if VaR increases by 20%+
            'concentration_limit': 0.25,  # Alert if single position >25%
            'correlation_spike': 0.8,  # Alert if correlation >0.8
            'volatility_spike_pct': 50.0  # Alert if volatility increases by 50%+
        }
        
        # Monitoring tasks
        self._monitoring_task = None
        self._alert_callbacks: List[Callable] = []
        
        # Risk calculation engines
        self.risk_calculator = None  # Will be injected
        
    async def start_monitoring(self, portfolio_ids: List[str]):
        """Start real-time risk monitoring for specified portfolios"""
        try:
            logger.info(f"Starting risk monitoring for portfolios: {portfolio_ids}")
            
            self.monitoring_active = True
            self.monitored_portfolios = portfolio_ids
            
            # Start the monitoring loop
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Risk monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Error starting risk monitoring: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop real-time risk monitoring"""
        try:
            logger.info("Stopping risk monitoring")
            
            self.monitoring_active = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Risk monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping risk monitoring: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop that runs every second"""
        try:
            while self.monitoring_active:
                start_time = datetime.utcnow()
                
                # Update risk metrics for all monitored portfolios
                for portfolio_id in self.monitored_portfolios:
                    try:
                        await self._update_portfolio_risk(portfolio_id)
                    except Exception as e:
                        logger.error(f"Error updating risk for portfolio {portfolio_id}: {e}")
                
                # Check for alerts
                await self._check_risk_alerts()
                
                # Calculate sleep time to maintain 1-second intervals
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Risk monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in risk monitoring loop: {e}")
            self.monitoring_active = False
    
    async def _update_portfolio_risk(self, portfolio_id: str):
        """Update risk metrics for a specific portfolio"""
        try:
            # Get current positions
            positions = await self._get_current_positions(portfolio_id)
            
            if not positions:
                return
            
            # Calculate current risk metrics
            risk_metrics = await self._calculate_risk_metrics(portfolio_id, positions)
            
            # Create snapshot
            snapshot = RiskMetricSnapshot(
                timestamp=datetime.utcnow(),
                portfolio_id=portfolio_id,
                total_exposure=risk_metrics.get('total_exposure', Decimal('0')),
                net_exposure=risk_metrics.get('net_exposure', Decimal('0')),
                var_1d_95=risk_metrics.get('var_1d_95', Decimal('0')),
                var_1d_99=risk_metrics.get('var_1d_99', Decimal('0')),
                portfolio_beta=risk_metrics.get('portfolio_beta', 0.0),
                portfolio_volatility=risk_metrics.get('portfolio_volatility', 0.0),
                max_position_concentration=risk_metrics.get('max_concentration', 0.0),
                correlation_risk_score=risk_metrics.get('correlation_risk', 0.0),
                liquidity_risk_score=risk_metrics.get('liquidity_risk', 0.0)
            )
            
            # Store snapshot
            self.risk_snapshots[portfolio_id].append(snapshot)
            
            # Keep only last 3600 snapshots (1 hour at 1-second intervals)
            if len(self.risk_snapshots[portfolio_id]) > 3600:
                self.risk_snapshots[portfolio_id] = self.risk_snapshots[portfolio_id][-3600:]
            
            # Send real-time update via WebSocket
            await self._broadcast_risk_update(snapshot)
            
        except Exception as e:
            logger.error(f"Error updating portfolio risk for {portfolio_id}: {e}")
            raise
    
    async def _get_current_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get current positions for a portfolio"""
        try:
            if self.position_service:
                return await self.position_service.get_positions(portfolio_id)
            else:
                # Mock positions for development
                return [
                    {
                        'symbol': 'AAPL',
                        'quantity': Decimal('100'),
                        'market_value': Decimal('15000'),
                        'unrealized_pnl': Decimal('500')
                    },
                    {
                        'symbol': 'GOOGL', 
                        'quantity': Decimal('50'),
                        'market_value': Decimal('12000'),
                        'unrealized_pnl': Decimal('-200')
                    }
                ]
        except Exception as e:
            logger.error(f"Error getting positions for {portfolio_id}: {e}")
            return []
    
    async def _calculate_risk_metrics(self, portfolio_id: str, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for current positions"""
        try:
            if not positions:
                return {}
            
            # Calculate basic exposure metrics
            total_exposure = sum(abs(pos['market_value']) for pos in positions)
            long_exposure = sum(pos['market_value'] for pos in positions if pos['market_value'] > 0)
            short_exposure = sum(pos['market_value'] for pos in positions if pos['market_value'] < 0)
            net_exposure = long_exposure + short_exposure
            
            # Calculate concentration risk
            max_concentration = 0.0
            if total_exposure > 0:
                max_concentration = max(abs(pos['market_value']) / total_exposure for pos in positions)
            
            # Mock advanced calculations (would integrate with risk_calculator in production)
            var_1d_95 = total_exposure * Decimal('0.02')  # 2% of total exposure
            var_1d_99 = total_exposure * Decimal('0.035')  # 3.5% of total exposure
            
            return {
                'total_exposure': Decimal(str(total_exposure)),
                'net_exposure': Decimal(str(net_exposure)),
                'var_1d_95': var_1d_95,
                'var_1d_99': var_1d_99,
                'portfolio_beta': 1.1,  # Mock value
                'portfolio_volatility': 0.18,  # Mock value
                'max_concentration': max_concentration,
                'correlation_risk': 0.3,  # Mock value
                'liquidity_risk': 0.2  # Mock value
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _check_risk_alerts(self):
        """Check for risk alerts across all monitored portfolios"""
        try:
            for portfolio_id in self.monitored_portfolios:
                snapshots = self.risk_snapshots.get(portfolio_id, [])
                
                if len(snapshots) < 2:
                    continue
                
                current = snapshots[-1]
                previous = snapshots[-2]
                
                # Check for VaR increase alerts
                if previous.var_1d_95 > 0:
                    var_increase_pct = float((current.var_1d_95 - previous.var_1d_95) / previous.var_1d_95 * 100)
                    if var_increase_pct > self.alert_thresholds['var_increase_pct']:
                        await self._trigger_alert(
                            portfolio_id, 
                            'var_spike',
                            f"VaR increased by {var_increase_pct:.1f}% in the last second",
                            'warning'
                        )
                
                # Check concentration limits
                if current.max_position_concentration > self.alert_thresholds['concentration_limit']:
                    await self._trigger_alert(
                        portfolio_id,
                        'concentration_breach',
                        f"Position concentration {current.max_position_concentration:.1%} exceeds limit",
                        'critical'
                    )
                
                # Check correlation risk
                if current.correlation_risk_score > self.alert_thresholds['correlation_spike']:
                    await self._trigger_alert(
                        portfolio_id,
                        'correlation_spike',
                        f"High correlation risk detected: {current.correlation_risk_score:.2f}",
                        'warning'
                    )
                
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    async def _trigger_alert(self, portfolio_id: str, alert_type: str, message: str, severity: str):
        """Trigger a risk alert"""
        try:
            alert = {
                'id': f"{portfolio_id}_{alert_type}_{int(datetime.utcnow().timestamp())}",
                'portfolio_id': portfolio_id,
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.utcnow().isoformat(),
                'acknowledged': False
            }
            
            # Notify all alert callbacks
            for callback in self._alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Send via WebSocket
            await self._broadcast_alert(alert)
            
            logger.warning(f"Risk alert triggered: {alert}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def _broadcast_risk_update(self, snapshot: RiskMetricSnapshot):
        """Broadcast risk update via WebSocket"""
        try:
            if self.websocket_manager:
                message = {
                    'type': 'risk_update',
                    'data': snapshot.to_dict()
                }
                await self.websocket_manager.broadcast_to_room(
                    f"risk_{snapshot.portfolio_id}", 
                    json.dumps(message)
                )
        except Exception as e:
            logger.error(f"Error broadcasting risk update: {e}")
    
    async def _broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast risk alert via WebSocket"""
        try:
            if self.websocket_manager:
                message = {
                    'type': 'risk_alert',
                    'data': alert
                }
                await self.websocket_manager.broadcast_to_room(
                    f"risk_{alert['portfolio_id']}", 
                    json.dumps(message)
                )
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for risk alerts"""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    async def get_current_risk_metrics(self, portfolio_id: str) -> Optional[RiskMetricSnapshot]:
        """Get current risk metrics for a portfolio"""
        snapshots = self.risk_snapshots.get(portfolio_id, [])
        return snapshots[-1] if snapshots else None
    
    async def get_risk_history(self, portfolio_id: str, hours: int = 1) -> List[RiskMetricSnapshot]:
        """Get risk history for a portfolio"""
        snapshots = self.risk_snapshots.get(portfolio_id, [])
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [s for s in snapshots if s.timestamp >= cutoff_time]
    
    async def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        logger.info(f"Updated alert thresholds: {self.alert_thresholds}")
    
    async def force_risk_calculation(self, portfolio_id: str):
        """Force immediate risk calculation for a portfolio"""
        try:
            await self._update_portfolio_risk(portfolio_id)
            logger.info(f"Forced risk calculation completed for portfolio {portfolio_id}")
        except Exception as e:
            logger.error(f"Error in forced risk calculation: {e}")
            raise
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'monitored_portfolios': getattr(self, 'monitored_portfolios', []),
            'update_interval': self.update_interval,
            'total_snapshots': sum(len(snapshots) for snapshots in self.risk_snapshots.values()),
            'alert_thresholds': self.alert_thresholds,
            'active_callbacks': len(self._alert_callbacks)
        }

# Global instance
risk_monitor = RiskMonitor()
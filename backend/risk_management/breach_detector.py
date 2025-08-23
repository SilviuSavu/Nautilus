"""
Risk Limit Breach Detection System
Real-time breach detection with alert generation and severity classification
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, asdict
import json
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class BreachSeverity(Enum):
    """Severity levels for risk breaches"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Channels for alert delivery"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    SLACK = "slack"
    TEAMS = "teams"

@dataclass
class BreachAlert:
    """Risk breach alert with routing information"""
    id: str
    breach_id: str
    portfolio_id: str
    limit_id: str
    limit_name: str
    severity: BreachSeverity
    message: str
    current_value: Decimal
    threshold_value: Decimal
    breach_percentage: float
    timestamp: datetime
    channels: List[AlertChannel]
    recipients: List[str]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    escalated: bool = False
    escalation_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['severity'] = self.severity.value
        result['channels'] = [c.value for c in self.channels]
        result['current_value'] = str(self.current_value)
        result['threshold_value'] = str(self.threshold_value)
        result['timestamp'] = self.timestamp.isoformat()
        if result['acknowledged_at']:
            result['acknowledged_at'] = result['acknowledged_at'].isoformat()
        return result

@dataclass
class BreachPattern:
    """Pattern detected in breach history"""
    pattern_type: str  # 'recurring', 'escalating', 'clustering'
    portfolio_id: str
    limit_type: str
    frequency: float  # breaches per hour
    trend: str  # 'increasing', 'decreasing', 'stable'
    confidence: float  # 0-1 confidence in pattern
    first_occurrence: datetime
    last_occurrence: datetime
    breach_count: int
    
class BreachDetector:
    """
    Advanced breach detection system with pattern recognition,
    severity classification, and intelligent alerting
    """
    
    def __init__(self, limit_engine=None, websocket_manager=None):
        self.limit_engine = limit_engine
        self.websocket_manager = websocket_manager
        
        # Alert management
        self.alerts: Dict[str, BreachAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)  # Keep last 10k alerts
        
        # Pattern detection
        self.breach_patterns: Dict[str, BreachPattern] = {}
        self.pattern_window_hours = 24  # Look back 24 hours for patterns
        
        # Escalation rules
        self.escalation_rules = {
            'time_based': {
                'warning_minutes': 5,    # Escalate warning after 5 min
                'critical_minutes': 2,   # Escalate critical after 2 min
                'emergency_minutes': 1   # Escalate emergency after 1 min
            },
            'repetition_based': {
                'warning_count': 3,      # Escalate after 3 warnings
                'critical_count': 2,     # Escalate after 2 critical
                'emergency_count': 1     # Escalate immediately
            }
        }
        
        # Alert channels and recipients
        self.alert_routing = {
            'default': {
                'channels': [AlertChannel.WEBSOCKET, AlertChannel.EMAIL],
                'recipients': ['risk-team@example.com']
            },
            'critical': {
                'channels': [AlertChannel.WEBSOCKET, AlertChannel.EMAIL, AlertChannel.SMS],
                'recipients': ['risk-manager@example.com', '+1234567890']
            },
            'emergency': {
                'channels': [AlertChannel.WEBSOCKET, AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.WEBHOOK],
                'recipients': ['cro@example.com', '+1234567890', 'emergency-webhook-url']
            }
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_breaches_detected': 0,
            'alerts_generated': 0,
            'patterns_detected': 0,
            'false_positives': 0,
            'escalations': 0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self._monitoring_task = None
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._pattern_callbacks: List[Callable] = []
    
    async def start_monitoring(self):
        """Start breach detection monitoring"""
        try:
            logger.info("Starting breach detection monitoring")
            
            self.monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Connect to limit engine if available
            if self.limit_engine:
                self.limit_engine.add_escalation_callback(self._handle_limit_breach)
            
            logger.info("Breach detection monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting breach detection: {e}")
            raise
    
    async def stop_monitoring(self):
        """Stop breach detection monitoring"""
        try:
            logger.info("Stopping breach detection monitoring")
            
            self.monitoring_active = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Breach detection monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping breach detection: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Main monitoring loop for breach detection"""
        try:
            while self.monitoring_active:
                start_time = datetime.utcnow()
                
                # Check for escalations
                await self._check_escalations()
                
                # Detect patterns
                await self._detect_patterns()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Sleep for monitoring interval
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                sleep_time = max(0, 30.0 - elapsed)  # Run every 30 seconds
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Breach detection monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in breach detection monitoring loop: {e}")
            self.monitoring_active = False
    
    async def _handle_limit_breach(self, breach_info: Dict[str, Any]):
        """Handle breach notification from limit engine"""
        try:
            breach_id = breach_info.get('id')
            
            # Determine severity
            severity = await self._classify_breach_severity(breach_info)
            
            # Generate alert
            alert = await self._generate_alert(breach_info, severity)
            
            # Store alert
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Route alert to appropriate channels
            await self._route_alert(alert)
            
            # Update statistics
            self.detection_stats['total_breaches_detected'] += 1
            self.detection_stats['alerts_generated'] += 1
            
            logger.info(f"Detected breach and generated alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error handling limit breach: {e}")
    
    async def _classify_breach_severity(self, breach_info: Dict[str, Any]) -> BreachSeverity:
        """Classify breach severity based on multiple factors"""
        try:
            current_value = Decimal(str(breach_info.get('current_value', 0)))
            threshold_value = Decimal(str(breach_info.get('threshold_value', 1)))
            
            # Calculate breach percentage
            if threshold_value > 0:
                breach_pct = float((current_value - threshold_value) / threshold_value * 100)
            else:
                breach_pct = 100
            
            # Base severity on breach percentage
            if breach_pct >= 100:  # 100%+ over limit
                base_severity = BreachSeverity.EMERGENCY
            elif breach_pct >= 50:  # 50-100% over limit
                base_severity = BreachSeverity.CRITICAL
            elif breach_pct >= 20:  # 20-50% over limit
                base_severity = BreachSeverity.HIGH
            elif breach_pct >= 5:   # 5-20% over limit
                base_severity = BreachSeverity.MEDIUM
            else:                   # <5% over limit
                base_severity = BreachSeverity.LOW
            
            # Adjust severity based on limit type
            limit_type = breach_info.get('limit_type', '')
            if limit_type in ['var', 'leverage', 'concentration']:
                # These are critical limit types - escalate
                if base_severity == BreachSeverity.MEDIUM:
                    base_severity = BreachSeverity.HIGH
                elif base_severity == BreachSeverity.HIGH:
                    base_severity = BreachSeverity.CRITICAL
            
            # Adjust based on recent breach history
            portfolio_id = breach_info.get('portfolio_id')
            recent_breaches = await self._get_recent_breaches(portfolio_id, hours=1)
            
            if len(recent_breaches) >= 3:
                # Multiple breaches in short time - escalate
                if base_severity in [BreachSeverity.LOW, BreachSeverity.MEDIUM]:
                    base_severity = BreachSeverity.HIGH
                elif base_severity == BreachSeverity.HIGH:
                    base_severity = BreachSeverity.CRITICAL
            
            return base_severity
            
        except Exception as e:
            logger.error(f"Error classifying breach severity: {e}")
            return BreachSeverity.MEDIUM
    
    async def _generate_alert(self, breach_info: Dict[str, Any], severity: BreachSeverity) -> BreachAlert:
        """Generate alert for a breach"""
        try:
            timestamp = datetime.utcnow()
            alert_id = f"alert_{breach_info.get('limit_id', 'unknown')}_{int(timestamp.timestamp())}"
            
            current_value = Decimal(str(breach_info.get('current_value', 0)))
            threshold_value = Decimal(str(breach_info.get('threshold_value', 1)))
            
            # Calculate breach percentage
            breach_pct = 0.0
            if threshold_value > 0:
                breach_pct = float((current_value - threshold_value) / threshold_value * 100)
            
            # Generate message
            limit_name = breach_info.get('limit_name', 'Unknown Limit')
            portfolio_id = breach_info.get('portfolio_id', 'Unknown Portfolio')
            
            message = f"Risk limit breach detected: {limit_name} for portfolio {portfolio_id}. "
            message += f"Current value: {current_value}, Threshold: {threshold_value} "
            message += f"({breach_pct:+.1f}% over limit)"
            
            # Get routing information based on severity
            routing = self._get_alert_routing(severity)
            
            alert = BreachAlert(
                id=alert_id,
                breach_id=breach_info.get('id', 'unknown'),
                portfolio_id=portfolio_id,
                limit_id=breach_info.get('limit_id', 'unknown'),
                limit_name=limit_name,
                severity=severity,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value,
                breach_percentage=breach_pct,
                timestamp=timestamp,
                channels=routing['channels'],
                recipients=routing['recipients']
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            raise
    
    def _get_alert_routing(self, severity: BreachSeverity) -> Dict[str, Any]:
        """Get alert routing based on severity"""
        if severity == BreachSeverity.EMERGENCY:
            return self.alert_routing.get('emergency', self.alert_routing['default'])
        elif severity in [BreachSeverity.CRITICAL, BreachSeverity.HIGH]:
            return self.alert_routing.get('critical', self.alert_routing['default'])
        else:
            return self.alert_routing['default']
    
    async def _route_alert(self, alert: BreachAlert):
        """Route alert to configured channels"""
        try:
            for channel in alert.channels:
                try:
                    if channel == AlertChannel.WEBSOCKET:
                        await self._send_websocket_alert(alert)
                    elif channel == AlertChannel.EMAIL:
                        await self._send_email_alert(alert)
                    elif channel == AlertChannel.SMS:
                        await self._send_sms_alert(alert)
                    elif channel == AlertChannel.WEBHOOK:
                        await self._send_webhook_alert(alert)
                    elif channel == AlertChannel.SLACK:
                        await self._send_slack_alert(alert)
                    elif channel == AlertChannel.TEAMS:
                        await self._send_teams_alert(alert)
                        
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")
            
        except Exception as e:
            logger.error(f"Error routing alert: {e}")
    
    async def _send_websocket_alert(self, alert: BreachAlert):
        """Send alert via WebSocket"""
        try:
            if self.websocket_manager:
                message = {
                    'type': 'risk_breach_alert',
                    'data': alert.to_dict()
                }
                
                # Send to portfolio-specific room
                await self.websocket_manager.broadcast_to_room(
                    f"risk_{alert.portfolio_id}",
                    json.dumps(message)
                )
                
                # Send to global risk room
                await self.websocket_manager.broadcast_to_room(
                    "risk_alerts",
                    json.dumps(message)
                )
                
        except Exception as e:
            logger.error(f"Error sending WebSocket alert: {e}")
    
    async def _send_email_alert(self, alert: BreachAlert):
        """Send alert via email"""
        # Mock implementation - would integrate with email service
        logger.info(f"Email alert sent for breach {alert.id} to {alert.recipients}")
    
    async def _send_sms_alert(self, alert: BreachAlert):
        """Send alert via SMS"""
        # Mock implementation - would integrate with SMS service
        logger.info(f"SMS alert sent for breach {alert.id} to {alert.recipients}")
    
    async def _send_webhook_alert(self, alert: BreachAlert):
        """Send alert via webhook"""
        # Mock implementation - would make HTTP POST to webhook URL
        logger.info(f"Webhook alert sent for breach {alert.id}")
    
    async def _send_slack_alert(self, alert: BreachAlert):
        """Send alert via Slack"""
        # Mock implementation - would integrate with Slack API
        logger.info(f"Slack alert sent for breach {alert.id}")
    
    async def _send_teams_alert(self, alert: BreachAlert):
        """Send alert via Microsoft Teams"""
        # Mock implementation - would integrate with Teams API
        logger.info(f"Teams alert sent for breach {alert.id}")
    
    async def _check_escalations(self):
        """Check for alerts that need escalation"""
        try:
            now = datetime.utcnow()
            
            for alert in self.alerts.values():
                if alert.acknowledged or alert.escalated:
                    continue
                
                # Check time-based escalation
                minutes_since_alert = (now - alert.timestamp).total_seconds() / 60
                
                escalation_threshold = self.escalation_rules['time_based'].get(
                    f"{alert.severity.value}_minutes", 
                    10  # Default 10 minutes
                )
                
                if minutes_since_alert >= escalation_threshold:
                    await self._escalate_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking escalations: {e}")
    
    async def _escalate_alert(self, alert: BreachAlert):
        """Escalate an alert to higher severity"""
        try:
            # Upgrade severity
            if alert.severity == BreachSeverity.LOW:
                new_severity = BreachSeverity.MEDIUM
            elif alert.severity == BreachSeverity.MEDIUM:
                new_severity = BreachSeverity.HIGH
            elif alert.severity == BreachSeverity.HIGH:
                new_severity = BreachSeverity.CRITICAL
            elif alert.severity == BreachSeverity.CRITICAL:
                new_severity = BreachSeverity.EMERGENCY
            else:
                return  # Already at max severity
            
            # Update alert
            alert.severity = new_severity
            alert.escalated = True
            alert.escalation_count += 1
            
            # Update routing
            routing = self._get_alert_routing(new_severity)
            alert.channels = routing['channels']
            alert.recipients = routing['recipients']
            
            # Update message
            alert.message += f" [ESCALATED to {new_severity.value.upper()}]"
            
            # Re-route with new severity
            await self._route_alert(alert)
            
            # Update statistics
            self.detection_stats['escalations'] += 1
            
            logger.warning(f"Escalated alert {alert.id} to {new_severity.value}")
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
    
    async def _detect_patterns(self):
        """Detect patterns in breach history"""
        try:
            # Get recent breaches grouped by portfolio and limit type
            cutoff_time = datetime.utcnow() - timedelta(hours=self.pattern_window_hours)
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp >= cutoff_time
            ]
            
            # Group by portfolio and limit type
            grouped_breaches = defaultdict(list)
            for alert in recent_alerts:
                key = f"{alert.portfolio_id}_{alert.limit_id}"
                grouped_breaches[key].append(alert)
            
            # Analyze each group for patterns
            for key, alerts in grouped_breaches.items():
                if len(alerts) >= 3:  # Need at least 3 points for pattern
                    pattern = await self._analyze_breach_pattern(key, alerts)
                    if pattern:
                        self.breach_patterns[key] = pattern
                        await self._notify_pattern_detected(pattern)
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    async def _analyze_breach_pattern(self, key: str, alerts: List[BreachAlert]) -> Optional[BreachPattern]:
        """Analyze alerts for patterns"""
        try:
            if len(alerts) < 3:
                return None
            
            # Sort by timestamp
            sorted_alerts = sorted(alerts, key=lambda x: x.timestamp)
            
            # Calculate frequency
            time_span = (sorted_alerts[-1].timestamp - sorted_alerts[0].timestamp).total_seconds() / 3600
            frequency = len(alerts) / max(time_span, 1)  # breaches per hour
            
            # Analyze trend in breach percentages
            breach_percentages = [alert.breach_percentage for alert in sorted_alerts]
            
            if len(breach_percentages) >= 2:
                # Simple linear trend
                x = np.arange(len(breach_percentages))
                slope = np.polyfit(x, breach_percentages, 1)[0]
                
                if slope > 1:
                    trend = 'increasing'
                elif slope < -1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Determine pattern type
            if frequency > 2:  # More than 2 breaches per hour
                pattern_type = 'clustering'
            elif trend == 'increasing':
                pattern_type = 'escalating'
            elif frequency > 0.5:  # More than 1 breach per 2 hours
                pattern_type = 'recurring'
            else:
                return None  # No significant pattern
            
            # Calculate confidence based on consistency
            confidence = min(0.9, frequency / 5 + len(alerts) / 10)
            
            portfolio_id, limit_id = key.split('_', 1)
            
            pattern = BreachPattern(
                pattern_type=pattern_type,
                portfolio_id=portfolio_id,
                limit_type=limit_id,
                frequency=frequency,
                trend=trend,
                confidence=confidence,
                first_occurrence=sorted_alerts[0].timestamp,
                last_occurrence=sorted_alerts[-1].timestamp,
                breach_count=len(alerts)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing breach pattern: {e}")
            return None
    
    async def _notify_pattern_detected(self, pattern: BreachPattern):
        """Notify about detected pattern"""
        try:
            # Notify pattern callbacks
            for callback in self._pattern_callbacks:
                try:
                    await callback(pattern)
                except Exception as e:
                    logger.error(f"Error in pattern callback: {e}")
            
            # Update statistics
            self.detection_stats['patterns_detected'] += 1
            
            logger.warning(f"Pattern detected: {pattern.pattern_type} for {pattern.portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error notifying pattern detection: {e}")
    
    async def _get_recent_breaches(self, portfolio_id: str, hours: int = 1) -> List[BreachAlert]:
        """Get recent breaches for a portfolio"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.portfolio_id == portfolio_id and alert.timestamp >= cutoff_time
        ]
    
    async def _cleanup_old_alerts(self):
        """Clean up old acknowledged alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Remove old acknowledged alerts
            old_alerts = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.acknowledged and alert.acknowledged_at and alert.acknowledged_at < cutoff_time
            ]
            
            for alert_id in old_alerts:
                del self.alerts[alert_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def get_active_alerts(self, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active (unacknowledged) alerts"""
        try:
            active_alerts = []
            
            for alert in self.alerts.values():
                if alert.acknowledged:
                    continue
                
                if portfolio_id and alert.portfolio_id != portfolio_id:
                    continue
                
                active_alerts.append(alert.to_dict())
            
            # Sort by severity and timestamp
            severity_order = {
                BreachSeverity.EMERGENCY: 0,
                BreachSeverity.CRITICAL: 1,
                BreachSeverity.HIGH: 2,
                BreachSeverity.MEDIUM: 3,
                BreachSeverity.LOW: 4
            }
            
            active_alerts.sort(key=lambda x: (
                severity_order.get(BreachSeverity(x['severity']), 5),
                x['timestamp']
            ))
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def get_breach_statistics(self, portfolio_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get breach statistics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            relevant_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time and (
                    portfolio_id is None or alert.portfolio_id == portfolio_id
                )
            ]
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in relevant_alerts:
                severity_counts[alert.severity.value] += 1
            
            # Count by limit type
            limit_type_counts = defaultdict(int)
            for alert in relevant_alerts:
                limit_type_counts[alert.limit_id] += 1
            
            return {
                'total_breaches': len(relevant_alerts),
                'severity_breakdown': dict(severity_counts),
                'limit_type_breakdown': dict(limit_type_counts),
                'patterns_detected': len(self.breach_patterns),
                'detection_stats': self.detection_stats,
                'time_window_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting breach statistics: {e}")
            return {}
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for new alerts"""
        self._alert_callbacks.append(callback)
    
    def add_pattern_callback(self, callback: Callable):
        """Add callback for pattern detection"""
        self._pattern_callbacks.append(callback)
    
    async def update_alert_routing(self, severity: str, routing_config: Dict[str, Any]):
        """Update alert routing configuration"""
        self.alert_routing[severity] = routing_config
        logger.info(f"Updated alert routing for {severity}")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get breach detector monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'active_alerts': len([a for a in self.alerts.values() if not a.acknowledged]),
            'total_alerts': len(self.alerts),
            'patterns_detected': len(self.breach_patterns),
            'detection_stats': self.detection_stats,
            'alert_callbacks': len(self._alert_callbacks),
            'pattern_callbacks': len(self._pattern_callbacks)
        }

# Global instance
breach_detector = BreachDetector()
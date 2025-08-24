"""
Production Monitoring Dashboard for M4 Max Optimized Nautilus Trading Platform
Comprehensive real-time monitoring, alerting, and performance optimization dashboard.

Features:
- Real-time M4 Max hardware monitoring
- Container performance tracking
- Trading performance analytics
- Automated alert management
- Performance optimization recommendations
- Production health status
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from prometheus_client import CollectorRegistry, Gauge, Counter, Info
from prometheus_client.exposition import generate_latest

from m4max_hardware_monitor import M4MaxHardwareMonitor, M4MaxMetrics
from container_performance_monitor import ContainerPerformanceMonitor
from trading_performance_monitor import TradingPerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SystemHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    duration_seconds: int
    description: str
    enabled: bool = True

@dataclass
class Alert:
    """Active alert"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    acknowledgment: Optional[str] = None

@dataclass
class SystemHealthSummary:
    """Overall system health summary"""
    overall_status: SystemHealthStatus
    performance_score: float
    active_alerts: int
    critical_alerts: int
    warning_alerts: int
    uptime_seconds: float
    last_restart: Optional[datetime]
    optimization_opportunities: List[str]
    recommendations: List[str]

@dataclass
class PerformanceOptimization:
    """Performance optimization recommendation"""
    component: str
    current_utilization: float
    recommended_action: str
    potential_improvement: str
    priority: str
    implementation_complexity: str

class ProductionMonitoringDashboard:
    """Comprehensive production monitoring dashboard with M4 Max optimization"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Initialize monitoring components
        self.m4max_monitor = M4MaxHardwareMonitor(redis_host, redis_port)
        self.container_monitor = ContainerPerformanceMonitor(redis_host, redis_port)
        self.trading_monitor = TradingPerformanceMonitor(redis_host, redis_port)
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Alert management
        self.alert_rules = self._setup_alert_rules()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # System tracking
        self.system_start_time = datetime.now()
        self.monitoring = False
        
        # Notification settings
        self.notification_settings = {
            'slack_webhook': None,
            'email_settings': None,
            'pagerduty_key': None
        }
        
        logger.info("Production Monitoring Dashboard initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for dashboard monitoring"""
        # System Health Metrics
        self.system_health_status = Gauge(
            'nautilus_system_health_status',
            'Overall system health status (0=down, 1=critical, 2=degraded, 3=healthy)',
            registry=self.registry
        )
        
        self.active_alerts_gauge = Gauge(
            'nautilus_active_alerts',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )
        
        self.system_uptime_gauge = Gauge(
            'nautilus_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # Performance Optimization Metrics
        self.optimization_opportunities_gauge = Gauge(
            'nautilus_optimization_opportunities',
            'Number of performance optimization opportunities',
            ['component'],
            registry=self.registry
        )
        
        # Alert Management Metrics
        self.alerts_fired_counter = Counter(
            'nautilus_alerts_fired_total',
            'Total alerts fired',
            ['severity', 'rule_name'],
            registry=self.registry
        )
        
        self.alerts_resolved_counter = Counter(
            'nautilus_alerts_resolved_total',
            'Total alerts resolved',
            ['severity', 'rule_name'],
            registry=self.registry
        )
        
        # Dashboard Performance
        self.dashboard_update_duration = Gauge(
            'nautilus_dashboard_update_duration_seconds',
            'Time taken to update dashboard',
            registry=self.registry
        )
        
        # System Info
        self.system_info = Info(
            'nautilus_system_info',
            'System information',
            registry=self.registry
        )
    
    def _setup_alert_rules(self) -> List[AlertRule]:
        """Setup default alert rules for M4 Max optimization"""
        return [
            # M4 Max Hardware Alerts
            AlertRule(
                name="M4MaxCPUHigh",
                condition="m4max_cpu_p_cores_usage_percent",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=60,
                description="M4 Max P-cores utilization is high"
            ),
            AlertRule(
                name="M4MaxCPUCritical",
                condition="m4max_cpu_p_cores_usage_percent",
                threshold=98.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=30,
                description="M4 Max P-cores utilization is critical"
            ),
            AlertRule(
                name="M4MaxMemoryHigh",
                condition="m4max_unified_memory_usage_gb",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,
                description="M4 Max unified memory usage is high"
            ),
            AlertRule(
                name="M4MaxGPUHigh",
                condition="m4max_gpu_utilization_percent",
                threshold=95.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=180,
                description="M4 Max GPU utilization is high"
            ),
            AlertRule(
                name="M4MaxNeuralEngineHigh",
                condition="m4max_neural_engine_utilization_percent",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,
                description="M4 Max Neural Engine utilization is high"
            ),
            AlertRule(
                name="M4MaxThermalWarning",
                condition="m4max_thermal_state",
                threshold=2.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=10,
                description="M4 Max thermal warning state"
            ),
            
            # Trading Performance Alerts
            AlertRule(
                name="OrderLatencyHigh",
                condition="order_execution_latency_p95",
                threshold=0.1,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,
                description="Order execution latency is high"
            ),
            AlertRule(
                name="PerformanceScoreLow",
                condition="nautilus_overall_performance_score",
                threshold=70.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300,
                description="Overall performance score is low"
            ),
            
            # Container Health Alerts
            AlertRule(
                name="ContainerDown",
                condition="nautilus_container_health_status",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=30,
                description="Container is unhealthy"
            ),
            AlertRule(
                name="ContainerMemoryHigh",
                condition="nautilus_container_memory_usage_percent",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,
                description="Container memory usage is high"
            ),
            
            # Engine Health Alerts
            AlertRule(
                name="EngineDown",
                condition="nautilus_engine_health_status",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=30,
                description="Engine is unhealthy"
            )
        ]
    
    def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate alert rules against current metrics"""
        new_alerts = []
        current_time = datetime.now()
        
        try:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = self._get_metric_value(metrics, rule.condition)
                if metric_value is None:
                    continue
                
                # Check if condition is met
                alert_triggered = False
                if rule.name.endswith("High") or rule.name.endswith("Critical"):
                    alert_triggered = metric_value > rule.threshold
                elif rule.name.endswith("Low") or rule.name.endswith("Down"):
                    alert_triggered = metric_value < rule.threshold
                
                alert_key = f"{rule.name}_{rule.condition}"
                
                if alert_triggered:
                    # Check if alert already exists
                    if alert_key not in self.active_alerts:
                        # Create new alert
                        alert = Alert(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"{rule.description}: {metric_value:.2f}",
                            timestamp=current_time
                        )
                        
                        self.active_alerts[alert_key] = alert
                        new_alerts.append(alert)
                        
                        # Update metrics
                        self.alerts_fired_counter.labels(
                            severity=rule.severity.value,
                            rule_name=rule.name
                        ).inc()
                        
                        logger.warning(f"Alert fired: {rule.name} - {alert.message}")
                        
                else:
                    # Resolve alert if it exists
                    if alert_key in self.active_alerts:
                        alert = self.active_alerts[alert_key]
                        alert.resolved = True
                        alert.resolved_timestamp = current_time
                        
                        # Move to history
                        self.alert_history.append(alert)
                        del self.active_alerts[alert_key]
                        
                        # Update metrics
                        self.alerts_resolved_counter.labels(
                            severity=alert.severity.value,
                            rule_name=alert.rule_name
                        ).inc()
                        
                        logger.info(f"Alert resolved: {alert.rule_name}")
                        
        except Exception as e:
            logger.error(f"Error evaluating alerts: {e}")
        
        return new_alerts
    
    def _get_metric_value(self, metrics: Dict[str, Any], condition: str) -> Optional[float]:
        """Get metric value from metrics dict"""
        try:
            # Handle nested metric paths
            if '.' in condition:
                parts = condition.split('.')
                value = metrics
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return None
                return float(value)
            else:
                if condition in metrics:
                    return float(metrics[condition])
                return None
        except (ValueError, TypeError, KeyError):
            return None
    
    def generate_performance_optimizations(self, 
                                         m4max_metrics: Optional[M4MaxMetrics],
                                         container_metrics: List[Any],
                                         trading_metrics: Dict[str, Any]) -> List[PerformanceOptimization]:
        """Generate performance optimization recommendations"""
        optimizations = []
        
        try:
            if m4max_metrics:
                # CPU optimization opportunities
                if m4max_metrics.cpu_p_cores_usage < 50 and m4max_metrics.cpu_e_cores_usage < 30:
                    optimizations.append(PerformanceOptimization(
                        component="M4 Max CPU",
                        current_utilization=m4max_metrics.cpu_p_cores_usage,
                        recommended_action="Increase workload parallelization",
                        potential_improvement="30-50% performance increase",
                        priority="Medium",
                        implementation_complexity="Low"
                    ))
                
                # GPU optimization opportunities
                if m4max_metrics.gpu_utilization_percent < 20:
                    optimizations.append(PerformanceOptimization(
                        component="M4 Max GPU",
                        current_utilization=m4max_metrics.gpu_utilization_percent,
                        recommended_action="Implement GPU-accelerated computations",
                        potential_improvement="10-20x performance for parallel tasks",
                        priority="High",
                        implementation_complexity="Medium"
                    ))
                
                # Neural Engine optimization
                if m4max_metrics.neural_engine_utilization_percent < 10:
                    ml_workload_present = trading_metrics.get('ml_metrics', [])
                    if ml_workload_present:
                        optimizations.append(PerformanceOptimization(
                            component="M4 Max Neural Engine",
                            current_utilization=m4max_metrics.neural_engine_utilization_percent,
                            recommended_action="Optimize ML models for Neural Engine",
                            potential_improvement="5-10x ML inference performance",
                            priority="High",
                            implementation_complexity="High"
                        ))
                
                # Memory bandwidth optimization
                if m4max_metrics.unified_memory_bandwidth_gbps < 100:
                    optimizations.append(PerformanceOptimization(
                        component="M4 Max Memory",
                        current_utilization=m4max_metrics.unified_memory_bandwidth_gbps,
                        recommended_action="Optimize memory access patterns",
                        potential_improvement="20-30% memory performance",
                        priority="Medium",
                        implementation_complexity="Medium"
                    ))
            
            # Container optimization opportunities
            underutilized_containers = [
                m for m in container_metrics 
                if hasattr(m, 'cpu_usage_percent') and m.cpu_usage_percent < 10
            ]
            
            if len(underutilized_containers) > 3:
                optimizations.append(PerformanceOptimization(
                    component="Container Resources",
                    current_utilization=len(underutilized_containers),
                    recommended_action="Consolidate or scale down underutilized containers",
                    potential_improvement="10-20% resource savings",
                    priority="Low",
                    implementation_complexity="Low"
                ))
                
        except Exception as e:
            logger.error(f"Error generating optimizations: {e}")
        
        return optimizations
    
    def calculate_system_health(self,
                              m4max_metrics: Optional[M4MaxMetrics],
                              container_metrics: List[Any],
                              trading_metrics: Dict[str, Any],
                              active_alerts: List[Alert]) -> SystemHealthSummary:
        """Calculate overall system health status"""
        try:
            # Determine overall status based on alerts
            critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
            warning_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.WARNING])
            
            if critical_alerts > 0:
                overall_status = SystemHealthStatus.CRITICAL
            elif warning_alerts > 3:
                overall_status = SystemHealthStatus.DEGRADED
            elif warning_alerts > 0:
                overall_status = SystemHealthStatus.DEGRADED
            else:
                overall_status = SystemHealthStatus.HEALTHY
            
            # Calculate performance score
            performance_score = trading_metrics.get('overall_score', 50.0)
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.system_start_time).total_seconds()
            
            # Generate optimization opportunities
            optimizations = self.generate_performance_optimizations(
                m4max_metrics, container_metrics, trading_metrics
            )
            optimization_opportunities = [opt.recommended_action for opt in optimizations]
            
            # Generate recommendations
            recommendations = []
            if m4max_metrics and m4max_metrics.thermal_state in ['serious', 'critical']:
                recommendations.append("Reduce system load to prevent thermal throttling")
            
            if performance_score < 70:
                recommendations.append("Review and optimize trading algorithms")
            
            if critical_alerts > 0:
                recommendations.append("Address critical alerts immediately")
            
            if len(optimization_opportunities) > 0:
                recommendations.append("Implement performance optimizations")
            
            return SystemHealthSummary(
                overall_status=overall_status,
                performance_score=performance_score,
                active_alerts=len(active_alerts),
                critical_alerts=critical_alerts,
                warning_alerts=warning_alerts,
                uptime_seconds=uptime_seconds,
                last_restart=self.system_start_time,
                optimization_opportunities=optimization_opportunities,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return SystemHealthSummary(
                overall_status=SystemHealthStatus.CRITICAL,
                performance_score=0.0,
                active_alerts=999,
                critical_alerts=999,
                warning_alerts=0,
                uptime_seconds=0.0,
                last_restart=None,
                optimization_opportunities=[],
                recommendations=["System health calculation failed - check monitoring system"]
            )
    
    async def send_alert_notification(self, alert: Alert):
        """Send alert notification via configured channels"""
        try:
            message = f"🚨 {alert.severity.value.upper()}: {alert.rule_name}\n{alert.message}\nTime: {alert.timestamp}"
            
            # Slack notification
            if self.notification_settings.get('slack_webhook'):
                await self._send_slack_notification(message, alert.severity)
            
            # Email notification
            if self.notification_settings.get('email_settings'):
                await self._send_email_notification(alert)
            
            # PagerDuty notification for critical alerts
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                if self.notification_settings.get('pagerduty_key'):
                    await self._send_pagerduty_notification(alert)
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def _send_slack_notification(self, message: str, severity: AlertSeverity):
        """Send Slack notification"""
        try:
            webhook_url = self.notification_settings['slack_webhook']
            
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map[severity],
                        "text": message,
                        "title": "Nautilus Trading Platform Alert",
                        "ts": int(time.time())
                    }
                ]
            }
            
            requests.post(webhook_url, json=payload, timeout=10)
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            settings = self.notification_settings['email_settings']
            
            msg = MIMEMultipart()
            msg['From'] = settings['from_email']
            msg['To'] = ', '.join(settings['to_emails'])
            msg['Subject'] = f"Nautilus Alert: {alert.severity.value.upper()} - {alert.rule_name}"
            
            body = f"""
            Alert: {alert.rule_name}
            Severity: {alert.severity.value.upper()}
            Message: {alert.message}
            Time: {alert.timestamp}
            
            Please investigate immediately if this is a critical alert.
            
            Nautilus Trading Platform Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'])
            server.starttls()
            server.login(settings['username'], settings['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics from all monitoring systems"""
        start_time = time.time()
        
        try:
            # Collect metrics concurrently
            m4max_task = asyncio.create_task(
                asyncio.to_thread(self.m4max_monitor.collect_metrics)
            )
            container_task = asyncio.create_task(
                self.container_monitor.collect_all_metrics()
            )
            trading_task = asyncio.create_task(
                self.trading_monitor.collect_all_trading_metrics()
            )
            
            m4max_metrics, (container_metrics, engine_health), trading_metrics = await asyncio.gather(
                m4max_task, container_task, trading_task
            )
            
            # Compile comprehensive metrics
            all_metrics = {
                'timestamp': datetime.now().isoformat(),
                'm4max_metrics': asdict(m4max_metrics) if m4max_metrics else None,
                'container_metrics': [asdict(m) for m in container_metrics],
                'engine_health': [asdict(h) for h in engine_health],
                'trading_metrics': trading_metrics,
                'collection_time_seconds': time.time() - start_time
            }
            
            # Evaluate alerts
            new_alerts = self.evaluate_alerts(all_metrics)
            
            # Send notifications for new alerts
            for alert in new_alerts:
                await self.send_alert_notification(alert)
            
            # Calculate system health
            system_health = self.calculate_system_health(
                m4max_metrics, container_metrics, trading_metrics, list(self.active_alerts.values())
            )
            
            all_metrics['system_health'] = asdict(system_health)
            all_metrics['active_alerts'] = [asdict(a) for a in self.active_alerts.values()]
            
            # Update Prometheus metrics
            self._update_dashboard_prometheus_metrics(system_health)
            
            # Store in Redis
            self.redis_client.set(
                "monitoring:dashboard:current",
                json.dumps(all_metrics, default=str),
                ex=300
            )
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error collecting dashboard metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _update_dashboard_prometheus_metrics(self, system_health: SystemHealthSummary):
        """Update Prometheus metrics for dashboard"""
        try:
            # System health status
            status_mapping = {
                SystemHealthStatus.DOWN: 0,
                SystemHealthStatus.CRITICAL: 1,
                SystemHealthStatus.DEGRADED: 2,
                SystemHealthStatus.HEALTHY: 3
            }
            self.system_health_status.set(status_mapping[system_health.overall_status])
            
            # Active alerts by severity
            self.active_alerts_gauge.labels(severity='critical').set(system_health.critical_alerts)
            self.active_alerts_gauge.labels(severity='warning').set(system_health.warning_alerts)
            self.active_alerts_gauge.labels(severity='info').set(
                system_health.active_alerts - system_health.critical_alerts - system_health.warning_alerts
            )
            
            # System uptime
            self.system_uptime_gauge.set(system_health.uptime_seconds)
            
            # Optimization opportunities
            self.optimization_opportunities_gauge.labels(component='system').set(
                len(system_health.optimization_opportunities)
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard Prometheus metrics: {e}")
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous production monitoring"""
        logger.info(f"Starting production monitoring dashboard (interval: {interval}s)")
        self.monitoring = True
        
        # Start component monitors
        monitor_tasks = [
            asyncio.create_task(self.m4max_monitor.start_monitoring(interval=5.0)),
            asyncio.create_task(self.container_monitor.start_monitoring(interval=30.0)),
            asyncio.create_task(self.trading_monitor.start_monitoring(interval=60.0))
        ]
        
        # Main dashboard monitoring loop
        dashboard_task = asyncio.create_task(self._dashboard_monitoring_loop(interval))
        
        try:
            await asyncio.gather(dashboard_task, *monitor_tasks)
        except asyncio.CancelledError:
            logger.info("Production monitoring cancelled")
            # Stop all monitors
            self.m4max_monitor.stop_monitoring()
            self.container_monitor.stop_monitoring()
            self.trading_monitor.stop_monitoring()
            self.monitoring = False
    
    async def _dashboard_monitoring_loop(self, interval: float):
        """Main dashboard monitoring loop"""
        while self.monitoring:
            try:
                start_time = time.time()
                
                await self.collect_all_metrics()
                
                collection_time = time.time() - start_time
                self.dashboard_update_duration.set(collection_time)
                
                logger.debug(f"Dashboard update completed in {collection_time:.3f}s")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Stopping production monitoring dashboard")
        self.monitoring = False
        self.m4max_monitor.stop_monitoring()
        self.container_monitor.stop_monitoring()
        self.trading_monitor.stop_monitoring()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics from all monitoring components"""
        try:
            dashboard_metrics = generate_latest(self.registry).decode('utf-8')
            m4max_metrics = self.m4max_monitor.get_prometheus_metrics()
            container_metrics = self.container_monitor.get_prometheus_metrics()
            trading_metrics = self.trading_monitor.get_prometheus_metrics()
            
            return f"{dashboard_metrics}\n{m4max_metrics}\n{container_metrics}\n{trading_metrics}"
        except Exception as e:
            logger.error(f"Error getting Prometheus metrics: {e}")
            return generate_latest(self.registry).decode('utf-8')
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get current dashboard summary"""
        try:
            current_data = self.redis_client.get("monitoring:dashboard:current")
            if current_data:
                return json.loads(current_data)
            return {'error': 'No current dashboard data available'}
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {'error': str(e)}
    
    def configure_notifications(self,
                              slack_webhook: Optional[str] = None,
                              email_settings: Optional[Dict] = None,
                              pagerduty_key: Optional[str] = None):
        """Configure notification channels"""
        if slack_webhook:
            self.notification_settings['slack_webhook'] = slack_webhook
        if email_settings:
            self.notification_settings['email_settings'] = email_settings
        if pagerduty_key:
            self.notification_settings['pagerduty_key'] = pagerduty_key
        
        logger.info("Notification settings updated")

# Example usage
if __name__ == "__main__":
    async def main():
        dashboard = ProductionMonitoringDashboard()
        
        # Configure notifications (optional)
        # dashboard.configure_notifications(
        #     slack_webhook="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        # )
        
        # Test single collection
        metrics = await dashboard.collect_all_metrics()
        
        print(f"\n=== Production Monitoring Dashboard Summary ===")
        if 'system_health' in metrics:
            health = metrics['system_health']
            print(f"Overall Status: {health['overall_status']}")
            print(f"Performance Score: {health['performance_score']:.1f}")
            print(f"Active Alerts: {health['active_alerts']} (Critical: {health['critical_alerts']}, Warning: {health['warning_alerts']})")
            print(f"Uptime: {health['uptime_seconds']:.0f} seconds")
            
            if health['optimization_opportunities']:
                print(f"\nOptimization Opportunities:")
                for opp in health['optimization_opportunities']:
                    print(f"  - {opp}")
            
            if health['recommendations']:
                print(f"\nRecommendations:")
                for rec in health['recommendations']:
                    print(f"  - {rec}")
        
        # Start monitoring for 180 seconds
        print(f"\n=== Starting production monitoring for 3 minutes ===")
        monitoring_task = asyncio.create_task(dashboard.start_monitoring(interval=15.0))
        await asyncio.sleep(180)
        dashboard.stop_monitoring()
        monitoring_task.cancel()
        
        print("Production Monitoring Dashboard test completed.")
    
    asyncio.run(main())
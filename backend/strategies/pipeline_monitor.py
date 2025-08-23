"""
Deployment Pipeline Monitoring Service
Tracks pipeline status, deployment metrics, failure detection, alerting, and performance monitoring
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    LOG = "log"


class MetricType(Enum):
    """Types of metrics to track"""
    PIPELINE_DURATION = "pipeline_duration"
    DEPLOYMENT_SUCCESS_RATE = "deployment_success_rate"
    TEST_SUCCESS_RATE = "test_success_rate"
    ROLLBACK_FREQUENCY = "rollback_frequency"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    PERFORMANCE_SCORE = "performance_score"


class PipelineStage(Enum):
    """Pipeline stages to monitor"""
    SOURCE_CONTROL = "source_control"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY_DEV = "deploy_dev"
    DEPLOY_STAGING = "deploy_staging"
    DEPLOY_PRODUCTION = "deploy_production"
    POST_DEPLOYMENT_VALIDATION = "post_deployment_validation"


class MonitoringAlert(BaseModel):
    """Monitoring alert"""
    alert_id: str
    strategy_id: str
    environment: Optional[str] = None
    severity: AlertSeverity
    title: str
    message: str
    details: Dict[str, Any] = {}
    
    # Timing
    created_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Notification
    channels: List[AlertChannel] = []
    notified: bool = False
    
    # Related entities
    deployment_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    test_suite_id: Optional[str] = None


class MetricValue(BaseModel):
    """Metric value with timestamp"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = {}


class PipelineMetrics(BaseModel):
    """Pipeline performance metrics"""
    strategy_id: str
    environment: Optional[str] = None
    
    # Timing metrics
    pipeline_duration_seconds: float = 0.0
    test_duration_seconds: float = 0.0
    deployment_duration_seconds: float = 0.0
    
    # Success metrics
    tests_passed: int = 0
    tests_failed: int = 0
    deployments_successful: int = 0
    deployments_failed: int = 0
    
    # Performance metrics
    performance_score: float = 0.0
    resource_cpu_percent: float = 0.0
    resource_memory_percent: float = 0.0
    
    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    
    # Timestamp
    collected_at: datetime


class PipelineStatus(BaseModel):
    """Current pipeline status"""
    pipeline_id: str
    strategy_id: str
    version: str
    
    # Stage tracking
    current_stage: PipelineStage
    completed_stages: List[PipelineStage] = []
    failed_stages: List[PipelineStage] = []
    
    # Overall status
    status: str  # running, completed, failed, cancelled
    progress_percent: float = 0.0
    
    # Timing
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    current_metrics: Optional[PipelineMetrics] = None
    stage_durations: Dict[str, float] = {}


class AlertRule(BaseModel):
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    enabled: bool = True
    
    # Conditions
    metric_type: MetricType
    threshold_value: float
    threshold_operator: str  # '>', '<', '>=', '<=', '==', '!='
    evaluation_window_minutes: int = 10
    
    # Alert configuration
    severity: AlertSeverity
    channels: List[AlertChannel] = []
    cooldown_minutes: int = 60
    
    # Targeting
    strategy_ids: Optional[List[str]] = None
    environments: Optional[List[str]] = None


class NotificationConfig(BaseModel):
    """Notification configuration"""
    email_smtp_host: Optional[str] = None
    email_smtp_port: Optional[int] = None
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = []
    
    webhook_urls: List[str] = []
    slack_webhook_url: Optional[str] = None
    teams_webhook_url: Optional[str] = None


class PipelineMonitor:
    """Deployment pipeline monitoring service"""
    
    def __init__(self,
                 strategy_tester=None,
                 deployment_manager=None,
                 version_control=None,
                 rollback_service=None):
        self.strategy_tester = strategy_tester
        self.deployment_manager = deployment_manager
        self.version_control = version_control
        self.rollback_service = rollback_service
        
        # State management
        self._pipeline_statuses: Dict[str, PipelineStatus] = {}
        self._metrics_history: Dict[str, List[PipelineMetrics]] = {}
        self._active_alerts: Dict[str, MonitoringAlert] = {}
        self._alert_history: Dict[str, MonitoringAlert] = {}
        self._alert_rules: Dict[str, AlertRule] = {}
        
        # Configuration
        self._notification_config = NotificationConfig()
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Callbacks for external monitoring
        self._metric_collectors: List[Callable] = []
        self._alert_handlers: List[Callable] = []
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_deployment_failure_rate",
                name="High Deployment Failure Rate",
                description="Alert when deployment failure rate exceeds threshold",
                metric_type=MetricType.DEPLOYMENT_SUCCESS_RATE,
                threshold_value=80.0,
                threshold_operator="<",
                evaluation_window_minutes=30,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="pipeline_duration_exceeded",
                name="Pipeline Duration Exceeded",
                description="Alert when pipeline takes too long",
                metric_type=MetricType.PIPELINE_DURATION,
                threshold_value=3600.0,  # 1 hour
                threshold_operator=">",
                evaluation_window_minutes=5,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="frequent_rollbacks",
                name="Frequent Rollbacks",
                description="Alert when rollback frequency is high",
                metric_type=MetricType.ROLLBACK_FREQUENCY,
                threshold_value=5.0,  # More than 5 rollbacks in window
                threshold_operator=">",
                evaluation_window_minutes=240,  # 4 hours
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK, AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="low_test_success_rate",
                name="Low Test Success Rate",
                description="Alert when test success rate drops",
                metric_type=MetricType.TEST_SUCCESS_RATE,
                threshold_value=90.0,
                threshold_operator="<",
                evaluation_window_minutes=60,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="high_resource_utilization",
                name="High Resource Utilization",
                description="Alert when resource utilization is high",
                metric_type=MetricType.RESOURCE_UTILIZATION,
                threshold_value=85.0,
                threshold_operator=">",
                evaluation_window_minutes=15,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG]
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.rule_id] = rule
    
    def configure_notifications(self, config: NotificationConfig):
        """Configure notification settings"""
        self._notification_config = config
        logger.info("Updated notification configuration")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update alert rule"""
        self._alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    async def start_monitoring(self, pipeline_id: str, strategy_id: str, version: str):
        """Start monitoring a pipeline"""
        
        pipeline_status = PipelineStatus(
            pipeline_id=pipeline_id,
            strategy_id=strategy_id,
            version=version,
            current_stage=PipelineStage.SOURCE_CONTROL,
            status="running",
            started_at=datetime.utcnow()
        )
        
        self._pipeline_statuses[pipeline_id] = pipeline_status
        
        # Start monitoring task
        task = asyncio.create_task(
            self._monitor_pipeline(pipeline_id)
        )
        self._monitoring_tasks[pipeline_id] = task
        
        logger.info(f"Started monitoring pipeline {pipeline_id} for strategy {strategy_id}")
    
    async def stop_monitoring(self, pipeline_id: str):
        """Stop monitoring a pipeline"""
        
        if pipeline_id in self._monitoring_tasks:
            self._monitoring_tasks[pipeline_id].cancel()
            del self._monitoring_tasks[pipeline_id]
        
        if pipeline_id in self._pipeline_statuses:
            pipeline_status = self._pipeline_statuses[pipeline_id]
            if pipeline_status.status == "running":
                pipeline_status.status = "cancelled"
                pipeline_status.completed_at = datetime.utcnow()
        
        logger.info(f"Stopped monitoring pipeline {pipeline_id}")
    
    async def _monitor_pipeline(self, pipeline_id: str):
        """Monitor pipeline execution"""
        
        pipeline_status = self._pipeline_statuses[pipeline_id]
        monitoring_interval = 30  # seconds
        
        try:
            while pipeline_status.status == "running":
                # Collect current metrics
                metrics = await self._collect_pipeline_metrics(pipeline_id)
                pipeline_status.current_metrics = metrics
                
                # Store metrics history
                key = f"{pipeline_status.strategy_id}_{pipeline_id}"
                if key not in self._metrics_history:
                    self._metrics_history[key] = []
                
                self._metrics_history[key].append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self._metrics_history[key] = [
                    m for m in self._metrics_history[key] 
                    if m.collected_at > cutoff_time
                ]
                
                # Update pipeline progress
                await self._update_pipeline_progress(pipeline_id)
                
                # Check alert rules
                await self._evaluate_alert_rules(pipeline_status, metrics)
                
                # Run custom metric collectors
                for collector in self._metric_collectors:
                    try:
                        await collector(pipeline_id, pipeline_status, metrics)
                    except Exception as e:
                        logger.error(f"Custom metric collector error: {str(e)}")
                
                await asyncio.sleep(monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Pipeline monitoring cancelled for {pipeline_id}")
        except Exception as e:
            logger.error(f"Pipeline monitoring error for {pipeline_id}: {str(e)}")
            
            # Create error alert
            await self._create_alert(
                strategy_id=pipeline_status.strategy_id,
                severity=AlertSeverity.ERROR,
                title=f"Pipeline Monitoring Error",
                message=f"Monitoring failed for pipeline {pipeline_id}: {str(e)}",
                pipeline_id=pipeline_id
            )
    
    async def _collect_pipeline_metrics(self, pipeline_id: str) -> PipelineMetrics:
        """Collect current pipeline metrics"""
        
        pipeline_status = self._pipeline_statuses[pipeline_id]
        
        # Initialize metrics
        metrics = PipelineMetrics(
            strategy_id=pipeline_status.strategy_id,
            collected_at=datetime.utcnow()
        )
        
        # Calculate pipeline duration
        if pipeline_status.started_at:
            duration = (datetime.utcnow() - pipeline_status.started_at).total_seconds()
            metrics.pipeline_duration_seconds = duration
        
        # Collect test metrics if available
        if self.strategy_tester:
            test_suites = self.strategy_tester.list_test_suites(
                strategy_id=pipeline_status.strategy_id
            )
            
            recent_suites = [
                s for s in test_suites
                if s.started_at > datetime.utcnow() - timedelta(hours=1)
            ]
            
            if recent_suites:
                passed_tests = sum(1 for s in recent_suites if s.overall_score >= 70)
                failed_tests = len(recent_suites) - passed_tests
                
                metrics.tests_passed = passed_tests
                metrics.tests_failed = failed_tests
                
                if recent_suites:
                    avg_duration = sum(
                        (s.completed_at - s.started_at).total_seconds() 
                        for s in recent_suites if s.completed_at
                    ) / len([s for s in recent_suites if s.completed_at])
                    metrics.test_duration_seconds = avg_duration
        
        # Collect deployment metrics if available
        if self.deployment_manager:
            deployments = self.deployment_manager.list_deployments(
                strategy_id=pipeline_status.strategy_id
            )
            
            recent_deployments = [
                d for d in deployments
                if d.deployed_at > datetime.utcnow() - timedelta(hours=1)
            ]
            
            if recent_deployments:
                successful = sum(1 for d in recent_deployments if d.status == "deployed")
                failed = len(recent_deployments) - successful
                
                metrics.deployments_successful = successful
                metrics.deployments_failed = failed
                
                if recent_deployments:
                    completed_deployments = [d for d in recent_deployments if d.completed_at]
                    if completed_deployments:
                        avg_duration = sum(
                            (d.completed_at - d.deployed_at).total_seconds()
                            for d in completed_deployments
                        ) / len(completed_deployments)
                        metrics.deployment_duration_seconds = avg_duration
        
        # Mock resource utilization (in production, get from system monitors)
        import random
        metrics.resource_cpu_percent = random.uniform(10, 90)
        metrics.resource_memory_percent = random.uniform(20, 80)
        
        # Calculate performance score
        test_score = 100 if metrics.tests_failed == 0 else max(0, 100 - metrics.tests_failed * 20)
        deployment_score = 100 if metrics.deployments_failed == 0 else max(0, 100 - metrics.deployments_failed * 30)
        resource_score = max(0, 100 - max(metrics.resource_cpu_percent, metrics.resource_memory_percent))
        
        metrics.performance_score = (test_score + deployment_score + resource_score) / 3
        
        return metrics
    
    async def _update_pipeline_progress(self, pipeline_id: str):
        """Update pipeline progress"""
        
        pipeline_status = self._pipeline_statuses[pipeline_id]
        
        # Mock progress calculation based on time and external services
        elapsed_minutes = (datetime.utcnow() - pipeline_status.started_at).total_seconds() / 60
        
        # Simulate stage progression
        stages = list(PipelineStage)
        expected_duration_per_stage = 15  # minutes
        
        current_stage_index = min(len(stages) - 1, int(elapsed_minutes / expected_duration_per_stage))
        
        if current_stage_index < len(stages):
            new_stage = stages[current_stage_index]
            
            if new_stage != pipeline_status.current_stage:
                # Mark previous stage as completed
                if pipeline_status.current_stage not in pipeline_status.completed_stages:
                    pipeline_status.completed_stages.append(pipeline_status.current_stage)
                
                pipeline_status.current_stage = new_stage
                
                logger.info(f"Pipeline {pipeline_id} progressed to stage {new_stage.value}")
        
        # Calculate progress percentage
        pipeline_status.progress_percent = (len(pipeline_status.completed_stages) / len(stages)) * 100
        
        # Check if pipeline is complete
        if len(pipeline_status.completed_stages) >= len(stages) - 1:
            pipeline_status.status = "completed"
            pipeline_status.completed_at = datetime.utcnow()
            pipeline_status.progress_percent = 100.0
            
            # Stop monitoring
            await self.stop_monitoring(pipeline_id)
    
    async def _evaluate_alert_rules(self, pipeline_status: PipelineStatus, metrics: PipelineMetrics):
        """Evaluate alert rules against current metrics"""
        
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this strategy/environment
            if rule.strategy_ids and pipeline_status.strategy_id not in rule.strategy_ids:
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(rule.metric_type, pipeline_status, metrics)
            if metric_value is None:
                continue
            
            # Evaluate threshold
            triggered = self._evaluate_threshold(
                metric_value, 
                rule.threshold_value, 
                rule.threshold_operator
            )
            
            if triggered:
                # Check cooldown
                if await self._check_alert_cooldown(rule):
                    await self._create_alert(
                        strategy_id=pipeline_status.strategy_id,
                        severity=rule.severity,
                        title=rule.name,
                        message=f"{rule.description}. Current value: {metric_value}, threshold: {rule.threshold_operator} {rule.threshold_value}",
                        details={
                            "rule_id": rule.rule_id,
                            "metric_type": rule.metric_type.value,
                            "metric_value": metric_value,
                            "threshold_value": rule.threshold_value,
                            "threshold_operator": rule.threshold_operator
                        },
                        pipeline_id=pipeline_status.pipeline_id,
                        channels=rule.channels
                    )
    
    def _get_metric_value(self, metric_type: MetricType, pipeline_status: PipelineStatus, metrics: PipelineMetrics) -> Optional[float]:
        """Get metric value based on type"""
        
        if metric_type == MetricType.PIPELINE_DURATION:
            return metrics.pipeline_duration_seconds
        elif metric_type == MetricType.DEPLOYMENT_SUCCESS_RATE:
            total = metrics.deployments_successful + metrics.deployments_failed
            return (metrics.deployments_successful / total * 100) if total > 0 else 100.0
        elif metric_type == MetricType.TEST_SUCCESS_RATE:
            total = metrics.tests_passed + metrics.tests_failed
            return (metrics.tests_passed / total * 100) if total > 0 else 100.0
        elif metric_type == MetricType.ERROR_RATE:
            return float(metrics.error_count)
        elif metric_type == MetricType.RESOURCE_UTILIZATION:
            return max(metrics.resource_cpu_percent, metrics.resource_memory_percent)
        elif metric_type == MetricType.PERFORMANCE_SCORE:
            return metrics.performance_score
        elif metric_type == MetricType.ROLLBACK_FREQUENCY:
            # Get rollback count from rollback service
            if self.rollback_service:
                rollbacks = self.rollback_service.list_rollback_history(
                    strategy_id=pipeline_status.strategy_id
                )
                recent_rollbacks = [
                    r for r in rollbacks
                    if r.started_at > datetime.utcnow() - timedelta(hours=4)
                ]
                return float(len(recent_rollbacks))
            return 0.0
        
        return None
    
    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition"""
        
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        
        return False
    
    async def _check_alert_cooldown(self, rule: AlertRule) -> bool:
        """Check if alert is in cooldown period"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=rule.cooldown_minutes)
        
        for alert in self._active_alerts.values():
            if (alert.details.get("rule_id") == rule.rule_id and 
                alert.created_at > cutoff_time):
                return False
        
        for alert in self._alert_history.values():
            if (alert.details.get("rule_id") == rule.rule_id and 
                alert.created_at > cutoff_time):
                return False
        
        return True
    
    async def _create_alert(self, 
                            strategy_id: str,
                            severity: AlertSeverity,
                            title: str,
                            message: str,
                            details: Dict[str, Any] = None,
                            environment: str = None,
                            pipeline_id: str = None,
                            deployment_id: str = None,
                            test_suite_id: str = None,
                            channels: List[AlertChannel] = None):
        """Create and send alert"""
        
        alert_id = str(uuid.uuid4())
        
        alert = MonitoringAlert(
            alert_id=alert_id,
            strategy_id=strategy_id,
            environment=environment,
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            created_at=datetime.utcnow(),
            pipeline_id=pipeline_id,
            deployment_id=deployment_id,
            test_suite_id=test_suite_id,
            channels=channels or [AlertChannel.LOG]
        )
        
        self._active_alerts[alert_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Run custom alert handlers
        for handler in self._alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Custom alert handler error: {str(e)}")
        
        logger.warning(f"Created alert {alert_id}: {title}")
    
    async def _send_alert_notifications(self, alert: MonitoringAlert):
        """Send alert notifications through configured channels"""
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.TEAMS:
                    await self._send_teams_alert(alert)
                elif channel == AlertChannel.LOG:
                    await self._send_log_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {str(e)}")
        
        alert.notified = True
    
    async def _send_email_alert(self, alert: MonitoringAlert):
        """Send email alert"""
        
        if not self._notification_config.email_recipients:
            return
        
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        body = f"""
Strategy: {alert.strategy_id}
Environment: {alert.environment or 'N/A'}
Severity: {alert.severity.value}
Time: {alert.created_at}

{alert.message}

Details: {json.dumps(alert.details, indent=2)}
"""
        
        # Mock email sending (in production, use proper SMTP)
        logger.info(f"Email alert sent: {subject}")
    
    async def _send_webhook_alert(self, alert: MonitoringAlert):
        """Send webhook alert"""
        
        payload = {
            "alert_id": alert.alert_id,
            "strategy_id": alert.strategy_id,
            "environment": alert.environment,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.created_at.isoformat(),
            "details": alert.details
        }
        
        # Mock webhook sending (in production, use HTTP client)
        logger.info(f"Webhook alert sent: {alert.title}")
    
    async def _send_slack_alert(self, alert: MonitoringAlert):
        """Send Slack alert"""
        
        emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:"
        }.get(alert.severity, ":exclamation:")
        
        message = f"{emoji} *{alert.title}*\n{alert.message}"
        
        # Mock Slack sending (in production, use Slack webhook)
        logger.info(f"Slack alert sent: {alert.title}")
    
    async def _send_teams_alert(self, alert: MonitoringAlert):
        """Send Teams alert"""
        
        # Mock Teams sending (in production, use Teams webhook)
        logger.info(f"Teams alert sent: {alert.title}")
    
    async def _send_log_alert(self, alert: MonitoringAlert):
        """Send log alert"""
        
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.message}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert"""
        
        alert = self._active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.resolved_at = datetime.utcnow()
        
        # Move to history
        self._alert_history[alert_id] = alert
        del self._active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def add_metric_collector(self, collector: Callable):
        """Add custom metric collector"""
        self._metric_collectors.append(collector)
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self._alert_handlers.append(handler)
    
    # Query Methods
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """Get pipeline status by ID"""
        return self._pipeline_statuses.get(pipeline_id)
    
    def list_pipeline_statuses(self, 
                               strategy_id: str = None,
                               status: str = None) -> List[PipelineStatus]:
        """List pipeline statuses"""
        
        statuses = list(self._pipeline_statuses.values())
        
        if strategy_id:
            statuses = [s for s in statuses if s.strategy_id == strategy_id]
        
        if status:
            statuses = [s for s in statuses if s.status == status]
        
        return sorted(statuses, key=lambda s: s.started_at, reverse=True)
    
    def get_pipeline_metrics(self, pipeline_id: str) -> List[PipelineMetrics]:
        """Get metrics history for pipeline"""
        
        pipeline_status = self._pipeline_statuses.get(pipeline_id)
        if not pipeline_status:
            return []
        
        key = f"{pipeline_status.strategy_id}_{pipeline_id}"
        return self._metrics_history.get(key, [])
    
    def get_strategy_metrics(self, strategy_id: str, hours: int = 24) -> List[PipelineMetrics]:
        """Get metrics for strategy across all pipelines"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        all_metrics = []
        
        for key, metrics_list in self._metrics_history.items():
            if key.startswith(f"{strategy_id}_"):
                recent_metrics = [m for m in metrics_list if m.collected_at > cutoff_time]
                all_metrics.extend(recent_metrics)
        
        return sorted(all_metrics, key=lambda m: m.collected_at, reverse=True)
    
    def list_active_alerts(self, 
                           strategy_id: str = None,
                           severity: AlertSeverity = None) -> List[MonitoringAlert]:
        """List active alerts"""
        
        alerts = list(self._active_alerts.values())
        
        if strategy_id:
            alerts = [a for a in alerts if a.strategy_id == strategy_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_history(self, 
                          strategy_id: str = None,
                          hours: int = 24) -> List[MonitoringAlert]:
        """Get alert history"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = list(self._alert_history.values()) + list(self._active_alerts.values())
        
        if strategy_id:
            alerts = [a for a in alerts if a.strategy_id == strategy_id]
        
        alerts = [a for a in alerts if a.created_at > cutoff_time]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring service statistics"""
        
        active_pipelines = len([s for s in self._pipeline_statuses.values() if s.status == "running"])
        completed_pipelines = len([s for s in self._pipeline_statuses.values() if s.status == "completed"])
        failed_pipelines = len([s for s in self._pipeline_statuses.values() if s.status == "failed"])
        
        active_alerts = len(self._active_alerts)
        
        # Alert severity breakdown
        alert_severity_stats = {}
        for severity in AlertSeverity:
            count = len([a for a in self._active_alerts.values() if a.severity == severity])
            alert_severity_stats[severity.value] = count
        
        # Recent metrics summary
        recent_metrics = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for metrics_list in self._metrics_history.values():
            recent = [m for m in metrics_list if m.collected_at > cutoff_time]
            recent_metrics.extend(recent)
        
        avg_performance_score = 0.0
        if recent_metrics:
            avg_performance_score = sum(m.performance_score for m in recent_metrics) / len(recent_metrics)
        
        return {
            "active_pipelines": active_pipelines,
            "completed_pipelines": completed_pipelines,
            "failed_pipelines": failed_pipelines,
            "success_rate": (completed_pipelines / (completed_pipelines + failed_pipelines) * 100) if (completed_pipelines + failed_pipelines) > 0 else 100,
            "active_alerts": active_alerts,
            "alert_severity_breakdown": alert_severity_stats,
            "average_performance_score": avg_performance_score,
            "monitoring_tasks": len(self._monitoring_tasks),
            "alert_rules": len(self._alert_rules),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
pipeline_monitor = PipelineMonitor()
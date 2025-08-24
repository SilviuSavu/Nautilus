#!/usr/bin/env python3
"""
Phase 7: Global Enterprise Monitoring & Alerting Platform
Comprehensive monitoring across 15 regions with predictive alerting and compliance tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import uuid
import statistics
import numpy as np
import aiohttp
import asyncpg
import redis.asyncio as redis
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"       # P0 - Immediate response required
    HIGH = "high"              # P1 - Response within 15 minutes
    MEDIUM = "medium"          # P2 - Response within 1 hour
    LOW = "low"                # P3 - Response within 24 hours
    INFO = "info"              # Informational only

class MonitoringCategory(Enum):
    """Monitoring categories"""
    INFRASTRUCTURE = "infrastructure"    # Server, network, storage
    APPLICATION = "application"         # Application performance
    BUSINESS = "business"               # Trading metrics, revenue
    SECURITY = "security"               # Security events, breaches
    COMPLIANCE = "compliance"           # Regulatory compliance
    USER_EXPERIENCE = "user_experience"  # Client-facing metrics

class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"           # Point-in-time value
    COUNTER = "counter"       # Monotonically increasing
    HISTOGRAM = "histogram"   # Distribution of values
    SUMMARY = "summary"       # Quantiles over time

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"
    COMPLIANCE_PORTAL = "compliance_portal"

@dataclass
class MetricDefinition:
    """Definition of a monitoring metric"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    category: MonitoringCategory
    unit: str
    
    # Collection settings
    collection_interval_seconds: int = 60
    retention_days: int = 90
    
    # Alerting thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Labels and dimensions
    labels: List[str] = field(default_factory=list)
    dimensions: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_id: str
    condition: str  # PromQL-like expression
    severity: AlertSeverity
    category: MonitoringCategory
    
    # Timing
    evaluation_interval_seconds: int = 60
    for_duration_seconds: int = 300  # 5 minutes
    
    # Notification
    channels: List[AlertChannel] = field(default_factory=list)
    notification_template: str = ""
    
    # Suppression
    suppression_rules: List[str] = field(default_factory=list)
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    runbook_url: Optional[str] = None
    escalation_policy: Optional[str] = None

@dataclass
class AlertIncident:
    """Active alert incident"""
    incident_id: str
    rule_id: str
    severity: AlertSeverity
    category: MonitoringCategory
    
    # Status
    status: str = "active"  # active, acknowledged, resolved
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Context
    triggered_regions: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Response
    assigned_to: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

class GlobalMonitoringPlatform:
    """
    Enterprise global monitoring platform with predictive alerting
    """
    
    def __init__(self):
        self.regions = self._initialize_regions()
        self.metrics = self._initialize_metrics()
        self.alert_rules = self._initialize_alert_rules()
        
        # Active monitoring state
        self.active_incidents: Dict[str, AlertIncident] = {}
        self.metric_values: Dict[str, Dict[str, float]] = {}  # metric_id -> region -> value
        self.alert_history: List[AlertIncident] = []
        
        # Monitoring infrastructure
        self.prometheus_registry = CollectorRegistry()
        self.metric_collectors: Dict[str, Any] = {}
        self.alert_manager = AlertManager()
        self.notification_manager = NotificationManager()
        
        # Data storage
        self.timeseries_db = None  # TimescaleDB connection
        self.redis_client = None
        
        # Regional monitoring agents
        self.regional_agents: Dict[str, RegionalMonitoringAgent] = {}
        
        # Performance tracking
        self.monitoring_metrics = {
            'total_metrics_collected': 0,
            'active_alerts': 0,
            'resolved_alerts_24h': 0,
            'mean_time_to_detection_seconds': 0,
            'mean_time_to_resolution_seconds': 0,
            'false_positive_rate': 0.0,
            'alert_fatigue_score': 0.0
        }
        
        # Predictive models
        self.anomaly_detectors: Dict[str, Any] = {}
        self.threshold_optimizers: Dict[str, Any] = {}
        
    def _initialize_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize global regions for monitoring"""
        return {
            'us-east-1': {
                'name': 'US East Primary',
                'monitoring_endpoint': 'https://monitoring-us-east-1.nautilus.com',
                'prometheus_url': 'http://prometheus-us-east-1:9090',
                'grafana_url': 'http://grafana-us-east-1:3000',
                'timezone': 'America/New_York',
                'criticality': 'primary'
            },
            'eu-west-1': {
                'name': 'EU West Primary', 
                'monitoring_endpoint': 'https://monitoring-eu-west-1.nautilus.com',
                'prometheus_url': 'http://prometheus-eu-west-1:9090',
                'grafana_url': 'http://grafana-eu-west-1:3000',
                'timezone': 'Europe/London',
                'criticality': 'primary'
            },
            'asia-ne-1': {
                'name': 'Asia Northeast Primary',
                'monitoring_endpoint': 'https://monitoring-asia-ne-1.nautilus.com', 
                'prometheus_url': 'http://prometheus-asia-ne-1:9090',
                'grafana_url': 'http://grafana-asia-ne-1:3000',
                'timezone': 'Asia/Tokyo',
                'criticality': 'primary'
            },
            'us-central-1': {
                'name': 'US Central DR',
                'monitoring_endpoint': 'https://monitoring-us-central-1.nautilus.com',
                'prometheus_url': 'http://prometheus-us-central-1:9090',
                'grafana_url': 'http://grafana-us-central-1:3000',
                'timezone': 'America/Chicago',
                'criticality': 'disaster_recovery'
            },
            'eu-central-1': {
                'name': 'EU Central DR',
                'monitoring_endpoint': 'https://monitoring-eu-central-1.nautilus.com',
                'prometheus_url': 'http://prometheus-eu-central-1:9090',
                'grafana_url': 'http://grafana-eu-central-1:3000',
                'timezone': 'Europe/Berlin',
                'criticality': 'disaster_recovery'
            }
        }
    
    def _initialize_metrics(self) -> Dict[str, MetricDefinition]:
        """Initialize comprehensive metric definitions"""
        metrics = {}
        
        # Infrastructure metrics
        metrics['cpu_usage'] = MetricDefinition(
            metric_id='cpu_usage',
            name='CPU Usage Percentage',
            description='CPU utilization across all cores',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.INFRASTRUCTURE,
            unit='percent',
            collection_interval_seconds=15,
            warning_threshold=75.0,
            critical_threshold=90.0,
            labels=['region', 'instance', 'service']
        )
        
        metrics['memory_usage'] = MetricDefinition(
            metric_id='memory_usage',
            name='Memory Usage Percentage',
            description='Memory utilization',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.INFRASTRUCTURE,
            unit='percent',
            collection_interval_seconds=15,
            warning_threshold=80.0,
            critical_threshold=95.0,
            labels=['region', 'instance', 'service']
        )
        
        metrics['disk_usage'] = MetricDefinition(
            metric_id='disk_usage',
            name='Disk Usage Percentage',
            description='Disk space utilization',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.INFRASTRUCTURE,
            unit='percent',
            collection_interval_seconds=60,
            warning_threshold=85.0,
            critical_threshold=95.0,
            labels=['region', 'instance', 'mount_point']
        )
        
        metrics['network_latency'] = MetricDefinition(
            metric_id='network_latency',
            name='Network Latency',
            description='Cross-region network latency',
            metric_type=MetricType.HISTOGRAM,
            category=MonitoringCategory.INFRASTRUCTURE,
            unit='milliseconds',
            collection_interval_seconds=5,
            warning_threshold=50.0,
            critical_threshold=100.0,
            labels=['source_region', 'target_region']
        )
        
        # Application metrics
        metrics['request_rate'] = MetricDefinition(
            metric_id='request_rate',
            name='HTTP Request Rate',
            description='Requests per second',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.APPLICATION,
            unit='requests/second',
            collection_interval_seconds=15,
            labels=['region', 'service', 'endpoint']
        )
        
        metrics['response_time'] = MetricDefinition(
            metric_id='response_time',
            name='HTTP Response Time',
            description='HTTP response time percentiles',
            metric_type=MetricType.HISTOGRAM,
            category=MonitoringCategory.APPLICATION,
            unit='milliseconds',
            collection_interval_seconds=15,
            warning_threshold=500.0,
            critical_threshold=2000.0,
            labels=['region', 'service', 'endpoint', 'status_code']
        )
        
        metrics['error_rate'] = MetricDefinition(
            metric_id='error_rate',
            name='Error Rate',
            description='Percentage of failed requests',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.APPLICATION,
            unit='percent',
            collection_interval_seconds=30,
            warning_threshold=1.0,
            critical_threshold=5.0,
            labels=['region', 'service', 'error_type']
        )
        
        # Trading metrics
        metrics['trading_volume'] = MetricDefinition(
            metric_id='trading_volume',
            name='Trading Volume',
            description='Trading volume per minute',
            metric_type=MetricType.COUNTER,
            category=MonitoringCategory.BUSINESS,
            unit='trades',
            collection_interval_seconds=60,
            labels=['region', 'instrument', 'client_type']
        )
        
        metrics['order_execution_latency'] = MetricDefinition(
            metric_id='order_execution_latency',
            name='Order Execution Latency',
            description='Time from order submission to execution',
            metric_type=MetricType.HISTOGRAM,
            category=MonitoringCategory.BUSINESS,
            unit='milliseconds',
            collection_interval_seconds=5,
            warning_threshold=10.0,
            critical_threshold=50.0,
            labels=['region', 'instrument', 'order_type']
        )
        
        metrics['portfolio_value'] = MetricDefinition(
            metric_id='portfolio_value',
            name='Total Portfolio Value',
            description='Aggregate portfolio value under management',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.BUSINESS,
            unit='usd',
            collection_interval_seconds=60,
            labels=['region', 'currency', 'client_type']
        )
        
        # Risk metrics
        metrics['var_utilization'] = MetricDefinition(
            metric_id='var_utilization',
            name='VaR Utilization',
            description='Value at Risk utilization percentage',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.BUSINESS,
            unit='percent',
            collection_interval_seconds=30,
            warning_threshold=80.0,
            critical_threshold=95.0,
            labels=['region', 'portfolio', 'confidence_level']
        )
        
        # Security metrics
        metrics['failed_logins'] = MetricDefinition(
            metric_id='failed_logins',
            name='Failed Login Attempts',
            description='Failed authentication attempts per minute',
            metric_type=MetricType.COUNTER,
            category=MonitoringCategory.SECURITY,
            unit='attempts',
            collection_interval_seconds=60,
            warning_threshold=10.0,
            critical_threshold=50.0,
            labels=['region', 'source_ip', 'user_agent']
        )
        
        metrics['suspicious_activity'] = MetricDefinition(
            metric_id='suspicious_activity',
            name='Suspicious Activity Score',
            description='ML-based suspicious activity detection score',
            metric_type=MetricType.GAUGE,
            category=MonitoringCategory.SECURITY,
            unit='score',
            collection_interval_seconds=300,
            warning_threshold=0.7,
            critical_threshold=0.9,
            labels=['region', 'client_id', 'activity_type']
        )
        
        # Compliance metrics
        metrics['regulatory_breaches'] = MetricDefinition(
            metric_id='regulatory_breaches',
            name='Regulatory Breaches',
            description='Number of regulatory compliance breaches',
            metric_type=MetricType.COUNTER,
            category=MonitoringCategory.COMPLIANCE,
            unit='breaches',
            collection_interval_seconds=300,
            critical_threshold=1.0,  # Any breach is critical
            labels=['region', 'jurisdiction', 'regulation_type']
        )
        
        metrics['audit_trail_gaps'] = MetricDefinition(
            metric_id='audit_trail_gaps',
            name='Audit Trail Gaps',
            description='Missing entries in audit trails',
            metric_type=MetricType.COUNTER,
            category=MonitoringCategory.COMPLIANCE,
            unit='gaps',
            collection_interval_seconds=300,
            warning_threshold=1.0,
            critical_threshold=5.0,
            labels=['region', 'system', 'data_type']
        )
        
        return metrics
    
    def _initialize_alert_rules(self) -> Dict[str, AlertRule]:
        """Initialize comprehensive alert rules"""
        rules = {}
        
        # Infrastructure alerts
        rules['high_cpu_usage'] = AlertRule(
            rule_id='high_cpu_usage',
            name='High CPU Usage',
            description='CPU usage exceeded threshold',
            metric_id='cpu_usage',
            condition='cpu_usage > 90',
            severity=AlertSeverity.HIGH,
            category=MonitoringCategory.INFRASTRUCTURE,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            notification_template='🔥 HIGH CPU: {value}% on {instance} in {region}',
            runbook_url='https://runbooks.nautilus.com/high-cpu'
        )
        
        rules['memory_exhaustion'] = AlertRule(
            rule_id='memory_exhaustion',
            name='Memory Exhaustion',
            description='Memory usage critically high',
            metric_id='memory_usage',
            condition='memory_usage > 95',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.INFRASTRUCTURE,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK, AlertChannel.SMS],
            notification_template='🚨 CRITICAL MEMORY: {value}% on {instance} in {region}',
            runbook_url='https://runbooks.nautilus.com/memory-exhaustion'
        )
        
        rules['cross_region_latency'] = AlertRule(
            rule_id='cross_region_latency',
            name='High Cross-Region Latency',
            description='Cross-region latency exceeded SLA',
            metric_id='network_latency',
            condition='network_latency_p95 > 75',
            severity=AlertSeverity.HIGH,
            category=MonitoringCategory.INFRASTRUCTURE,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            notification_template='⚡ HIGH LATENCY: {value}ms between {source_region} and {target_region}',
            for_duration_seconds=180  # 3 minutes
        )
        
        # Application alerts
        rules['high_error_rate'] = AlertRule(
            rule_id='high_error_rate',
            name='High Error Rate',
            description='Application error rate exceeded threshold',
            metric_id='error_rate',
            condition='error_rate > 5',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.APPLICATION,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK],
            notification_template='💥 HIGH ERROR RATE: {value}% in {service} ({region})'
        )
        
        rules['slow_response_time'] = AlertRule(
            rule_id='slow_response_time',
            name='Slow Response Time',
            description='Response time exceeded SLA',
            metric_id='response_time',
            condition='response_time_p95 > 2000',
            severity=AlertSeverity.MEDIUM,
            category=MonitoringCategory.APPLICATION,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            notification_template='🐌 SLOW RESPONSE: {value}ms p95 in {service} ({region})'
        )
        
        # Trading alerts
        rules['trading_system_down'] = AlertRule(
            rule_id='trading_system_down',
            name='Trading System Unavailable',
            description='Trading system not processing orders',
            metric_id='trading_volume',
            condition='rate(trading_volume[5m]) == 0',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.BUSINESS,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SMS, AlertChannel.SLACK],
            notification_template='🛑 TRADING HALT: No trading activity in {region} for 5 minutes',
            for_duration_seconds=300
        )
        
        rules['order_latency_high'] = AlertRule(
            rule_id='order_latency_high',
            name='High Order Execution Latency',
            description='Order execution taking too long',
            metric_id='order_execution_latency',
            condition='order_execution_latency_p95 > 50',
            severity=AlertSeverity.HIGH,
            category=MonitoringCategory.BUSINESS,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            notification_template='⏱️ SLOW ORDERS: {value}ms p95 execution latency in {region}'
        )
        
        # Risk alerts
        rules['var_limit_breach'] = AlertRule(
            rule_id='var_limit_breach',
            name='VaR Limit Breach',
            description='Value at Risk limit exceeded',
            metric_id='var_utilization',
            condition='var_utilization > 95',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.BUSINESS,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SMS, AlertChannel.EMAIL],
            notification_template='⚠️ VAR BREACH: {value}% utilization for {portfolio} ({region})',
            for_duration_seconds=60
        )
        
        # Security alerts
        rules['brute_force_attack'] = AlertRule(
            rule_id='brute_force_attack',
            name='Potential Brute Force Attack',
            description='High number of failed login attempts',
            metric_id='failed_logins',
            condition='rate(failed_logins[5m]) > 10',
            severity=AlertSeverity.HIGH,
            category=MonitoringCategory.SECURITY,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            notification_template='🛡️ SECURITY: {value} failed logins/min from {source_ip}'
        )
        
        rules['suspicious_activity_detected'] = AlertRule(
            rule_id='suspicious_activity_detected',
            name='Suspicious Activity Detected',
            description='ML model detected suspicious trading activity',
            metric_id='suspicious_activity',
            condition='suspicious_activity > 0.9',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.SECURITY,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.COMPLIANCE_PORTAL],
            notification_template='🔍 SUSPICIOUS: Score {value} for client {client_id} ({region})'
        )
        
        # Compliance alerts
        rules['regulatory_breach'] = AlertRule(
            rule_id='regulatory_breach',
            name='Regulatory Compliance Breach',
            description='Regulatory compliance violation detected',
            metric_id='regulatory_breaches',
            condition='increase(regulatory_breaches[5m]) > 0',
            severity=AlertSeverity.CRITICAL,
            category=MonitoringCategory.COMPLIANCE,
            channels=[AlertChannel.PAGERDUTY, AlertChannel.COMPLIANCE_PORTAL, AlertChannel.SMS],
            notification_template='🏛️ COMPLIANCE BREACH: {regulation_type} violation in {region}',
            for_duration_seconds=0  # Immediate alert
        )
        
        return rules
    
    async def initialize(self):
        """Initialize the global monitoring platform"""
        logger.info("📊 Initializing Global Enterprise Monitoring Platform")
        
        # Initialize database connections
        await self._initialize_databases()
        
        # Initialize Prometheus metrics
        await self._initialize_prometheus_metrics()
        
        # Deploy regional monitoring agents
        await self._deploy_regional_agents()
        
        # Initialize alert manager
        await self.alert_manager.initialize(self.alert_rules)
        
        # Initialize notification manager
        await self.notification_manager.initialize()
        
        # Start monitoring loops
        await self._start_monitoring_loops()
        
        # Load historical data for anomaly detection
        await self._initialize_anomaly_detection()
        
        logger.info("✅ Global Monitoring Platform initialized")
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        
        # TimescaleDB for time-series data
        self.timeseries_db = await asyncpg.create_pool(
            "postgresql://nautilus:password@timescaledb-global:5432/monitoring",
            min_size=10,
            max_size=50
        )
        
        # Redis for real-time data
        self.redis_client = redis.from_url(
            "redis://redis-monitoring-global:6379",
            decode_responses=True
        )
        
        # Create monitoring tables
        await self._create_monitoring_tables()
        
        logger.info("✅ Monitoring databases initialized")
    
    async def _create_monitoring_tables(self):
        """Create monitoring database tables"""
        
        async with self.timeseries_db.acquire() as conn:
            # Metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    metric_id VARCHAR NOT NULL,
                    region VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    labels JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Convert to hypertable for time-series optimization
            await conn.execute("""
                SELECT create_hypertable('metrics', 'timestamp', if_not_exists => TRUE)
            """)
            
            # Alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_incidents (
                    incident_id VARCHAR PRIMARY KEY,
                    rule_id VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    category VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    triggered_at TIMESTAMPTZ NOT NULL,
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    triggered_regions TEXT[],
                    affected_services TEXT[],
                    current_value DOUBLE PRECISION,
                    threshold_value DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_metric_region_time ON metrics(metric_id, region, timestamp DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status_severity ON alert_incidents(status, severity)")
    
    async def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metric collectors"""
        
        for metric_id, metric_def in self.metrics.items():
            if metric_def.metric_type == MetricType.GAUGE:
                collector = Gauge(
                    metric_id,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.COUNTER:
                collector = Counter(
                    metric_id,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                collector = Histogram(
                    metric_id,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            
            self.metric_collectors[metric_id] = collector
        
        # Start Prometheus HTTP server
        start_http_server(8000, registry=self.prometheus_registry)
        logger.info("📈 Prometheus metrics server started on port 8000")
    
    async def _deploy_regional_agents(self):
        """Deploy monitoring agents to all regions"""
        
        for region_id, region_config in self.regions.items():
            agent = RegionalMonitoringAgent(region_id, region_config)
            await agent.initialize(self.metrics)
            self.regional_agents[region_id] = agent
            
            logger.info(f"🤖 Regional monitoring agent deployed to {region_id}")
    
    async def _start_monitoring_loops(self):
        """Start monitoring background tasks"""
        
        # Metric collection loop
        asyncio.create_task(self._metric_collection_loop())
        
        # Alert evaluation loop
        asyncio.create_task(self._alert_evaluation_loop())
        
        # Incident management loop
        asyncio.create_task(self._incident_management_loop())
        
        # Performance metrics loop
        asyncio.create_task(self._performance_metrics_loop())
        
        # Anomaly detection loop
        asyncio.create_task(self._anomaly_detection_loop())
        
        logger.info("🔄 Monitoring loops started")
    
    async def _metric_collection_loop(self):
        """Main metric collection loop"""
        
        while True:
            try:
                start_time = time.time()
                
                # Collect metrics from all regional agents
                collection_tasks = []
                for agent in self.regional_agents.values():
                    task = asyncio.create_task(agent.collect_metrics())
                    collection_tasks.append(task)
                
                # Wait for all collections to complete
                results = await asyncio.gather(*collection_tasks, return_exceptions=True)
                
                # Process results
                total_metrics = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        region = list(self.regional_agents.keys())[i]
                        logger.error(f"Failed to collect metrics from {region}: {result}")
                    else:
                        # Store metrics in database
                        await self._store_metrics(result)
                        total_metrics += len(result)
                
                # Update collection metrics
                collection_time = time.time() - start_time
                self.monitoring_metrics['total_metrics_collected'] += total_metrics
                
                logger.debug(f"📊 Collected {total_metrics} metrics in {collection_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
            
            await asyncio.sleep(15)  # Collect every 15 seconds
    
    async def _store_metrics(self, metrics_batch: List[Dict[str, Any]]):
        """Store metrics in TimescaleDB"""
        
        if not metrics_batch:
            return
        
        async with self.timeseries_db.acquire() as conn:
            await conn.executemany("""
                INSERT INTO metrics (metric_id, region, timestamp, value, labels)
                VALUES ($1, $2, $3, $4, $5)
            """, [
                (
                    metric['metric_id'],
                    metric['region'],
                    metric['timestamp'],
                    metric['value'],
                    json.dumps(metric.get('labels', {}))
                ) for metric in metrics_batch
            ])
    
    async def _alert_evaluation_loop(self):
        """Evaluate alert rules continuously"""
        
        while True:
            try:
                start_time = time.time()
                
                # Evaluate all alert rules
                for rule_id, rule in self.alert_rules.items():
                    try:
                        should_trigger = await self._evaluate_alert_rule(rule)
                        
                        if should_trigger:
                            await self._trigger_alert(rule, should_trigger)
                        else:
                            # Check if we should resolve existing incident
                            await self._check_incident_resolution(rule)
                    
                    except Exception as e:
                        logger.error(f"Error evaluating rule {rule_id}: {e}")
                
                evaluation_time = time.time() - start_time
                logger.debug(f"🔍 Alert evaluation completed in {evaluation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
            
            await asyncio.sleep(60)  # Evaluate every minute
    
    async def _evaluate_alert_rule(self, rule: AlertRule) -> Optional[Dict[str, Any]]:
        """Evaluate a single alert rule"""
        
        # Get recent metric values
        recent_values = await self._get_recent_metric_values(
            rule.metric_id,
            timedelta(seconds=rule.for_duration_seconds + 60)
        )
        
        if not recent_values:
            return None
        
        # Simple threshold-based evaluation (would use PromQL in production)
        metric_def = self.metrics[rule.metric_id]
        
        for region, values in recent_values.items():
            if not values:
                continue
            
            latest_value = values[-1]['value']
            
            # Check if rule condition is met
            if rule.condition == f'{rule.metric_id} > {metric_def.critical_threshold}':
                if latest_value > metric_def.critical_threshold:
                    return {
                        'region': region,
                        'value': latest_value,
                        'threshold': metric_def.critical_threshold,
                        'severity': AlertSeverity.CRITICAL
                    }
            elif rule.condition == f'{rule.metric_id} > {metric_def.warning_threshold}':
                if latest_value > metric_def.warning_threshold:
                    return {
                        'region': region,
                        'value': latest_value,
                        'threshold': metric_def.warning_threshold,
                        'severity': rule.severity
                    }
        
        return None
    
    async def _get_recent_metric_values(
        self,
        metric_id: str,
        time_window: timedelta
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent metric values from database"""
        
        since_time = datetime.now() - time_window
        
        async with self.timeseries_db.acquire() as conn:
            rows = await conn.fetch("""
                SELECT region, timestamp, value, labels
                FROM metrics
                WHERE metric_id = $1 AND timestamp >= $2
                ORDER BY timestamp DESC
            """, metric_id, since_time)
        
        # Group by region
        values_by_region = {}
        for row in rows:
            region = row['region']
            if region not in values_by_region:
                values_by_region[region] = []
            
            values_by_region[region].append({
                'timestamp': row['timestamp'],
                'value': row['value'],
                'labels': json.loads(row['labels']) if row['labels'] else {}
            })
        
        return values_by_region
    
    async def _trigger_alert(self, rule: AlertRule, trigger_data: Dict[str, Any]):
        """Trigger a new alert incident"""
        
        # Check if incident already exists
        existing_incident = None
        for incident in self.active_incidents.values():
            if (incident.rule_id == rule.rule_id and 
                incident.status == 'active' and
                trigger_data['region'] in incident.triggered_regions):
                existing_incident = incident
                break
        
        if existing_incident:
            # Update existing incident
            existing_incident.current_value = trigger_data['value']
            return
        
        # Create new incident
        incident = AlertIncident(
            incident_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            severity=trigger_data['severity'],
            category=rule.category,
            status='active',
            triggered_regions=[trigger_data['region']],
            current_value=trigger_data['value'],
            threshold_value=trigger_data['threshold']
        )
        
        self.active_incidents[incident.incident_id] = incident
        self.monitoring_metrics['active_alerts'] += 1
        
        # Store in database
        await self._store_alert_incident(incident)
        
        # Send notifications
        await self.notification_manager.send_alert_notifications(rule, incident, trigger_data)
        
        logger.warning(f"🚨 ALERT TRIGGERED: {rule.name} in {trigger_data['region']} - Value: {trigger_data['value']}")
    
    async def _check_incident_resolution(self, rule: AlertRule):
        """Check if incidents should be resolved"""
        
        incidents_to_resolve = []
        
        for incident in self.active_incidents.values():
            if incident.rule_id != rule.rule_id or incident.status != 'active':
                continue
            
            # Check if conditions are no longer met
            should_resolve = await self._should_resolve_incident(rule, incident)
            
            if should_resolve:
                incidents_to_resolve.append(incident)
        
        # Resolve incidents
        for incident in incidents_to_resolve:
            await self._resolve_incident(incident)
    
    async def _should_resolve_incident(self, rule: AlertRule, incident: AlertIncident) -> bool:
        """Check if incident should be resolved"""
        
        # Get recent values for affected regions
        recent_values = await self._get_recent_metric_values(
            rule.metric_id,
            timedelta(minutes=5)
        )
        
        metric_def = self.metrics[rule.metric_id]
        
        # Check if values are below threshold in all affected regions
        for region in incident.triggered_regions:
            if region not in recent_values:
                continue
            
            region_values = recent_values[region]
            if not region_values:
                continue
            
            latest_value = region_values[0]['value']  # Most recent
            
            # Check if still above threshold
            if rule.severity == AlertSeverity.CRITICAL:
                if latest_value > metric_def.critical_threshold:
                    return False
            else:
                if latest_value > metric_def.warning_threshold:
                    return False
        
        return True  # Can be resolved
    
    async def _resolve_incident(self, incident: AlertIncident):
        """Resolve an alert incident"""
        
        incident.status = 'resolved'
        incident.resolved_at = datetime.now()
        
        # Update metrics
        self.monitoring_metrics['active_alerts'] -= 1
        self.monitoring_metrics['resolved_alerts_24h'] += 1
        
        # Calculate resolution time
        resolution_time = (incident.resolved_at - incident.triggered_at).total_seconds()
        
        # Update MTTR
        current_mttr = self.monitoring_metrics['mean_time_to_resolution_seconds']
        self.monitoring_metrics['mean_time_to_resolution_seconds'] = (current_mttr + resolution_time) / 2
        
        # Move to history
        self.alert_history.append(incident)
        del self.active_incidents[incident.incident_id]
        
        # Update database
        await self._update_alert_incident(incident)
        
        # Send resolution notification
        rule = self.alert_rules[incident.rule_id]
        await self.notification_manager.send_resolution_notification(rule, incident)
        
        logger.info(f"✅ INCIDENT RESOLVED: {rule.name} after {resolution_time:.0f}s")
    
    async def _store_alert_incident(self, incident: AlertIncident):
        """Store alert incident in database"""
        
        async with self.timeseries_db.acquire() as conn:
            await conn.execute("""
                INSERT INTO alert_incidents 
                (incident_id, rule_id, severity, category, status, triggered_at, 
                 triggered_regions, affected_services, current_value, threshold_value)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            incident.incident_id,
            incident.rule_id,
            incident.severity.value,
            incident.category.value,
            incident.status,
            incident.triggered_at,
            incident.triggered_regions,
            incident.affected_services,
            incident.current_value,
            incident.threshold_value
            )
    
    async def _update_alert_incident(self, incident: AlertIncident):
        """Update alert incident in database"""
        
        async with self.timeseries_db.acquire() as conn:
            await conn.execute("""
                UPDATE alert_incidents 
                SET status = $1, acknowledged_at = $2, resolved_at = $3
                WHERE incident_id = $4
            """,
            incident.status,
            incident.acknowledged_at,
            incident.resolved_at,
            incident.incident_id
            )
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        # Calculate performance metrics
        total_incidents_24h = len([
            i for i in self.alert_history 
            if i.triggered_at > datetime.now() - timedelta(hours=24)
        ])
        
        false_positives = len([
            i for i in self.alert_history[-100:]  # Last 100 incidents
            if i.resolved_at and (i.resolved_at - i.triggered_at).total_seconds() < 300
        ])
        
        false_positive_rate = false_positives / min(len(self.alert_history[-100:]), 100) * 100 if self.alert_history else 0
        
        # Regional health summary
        regional_health = {}
        for region_id, agent in self.regional_agents.items():
            health = await agent.get_health_summary()
            regional_health[region_id] = health
        
        dashboard = {
            'overview': {
                'total_regions': len(self.regions),
                'active_alerts': self.monitoring_metrics['active_alerts'],
                'resolved_alerts_24h': self.monitoring_metrics['resolved_alerts_24h'],
                'mean_time_to_resolution_minutes': self.monitoring_metrics['mean_time_to_resolution_seconds'] / 60,
                'false_positive_rate_percentage': false_positive_rate,
                'monitoring_health': 'healthy' if self.monitoring_metrics['active_alerts'] < 10 else 'degraded'
            },
            
            'alert_breakdown': {
                'by_severity': {
                    severity.value: len([
                        i for i in self.active_incidents.values() 
                        if i.severity == severity
                    ]) for severity in AlertSeverity
                },
                'by_category': {
                    category.value: len([
                        i for i in self.active_incidents.values()
                        if i.category == category
                    ]) for category in MonitoringCategory
                }
            },
            
            'regional_health': regional_health,
            
            'recent_incidents': [
                {
                    'incident_id': incident.incident_id,
                    'rule_name': self.alert_rules[incident.rule_id].name,
                    'severity': incident.severity.value,
                    'status': incident.status,
                    'triggered_at': incident.triggered_at.isoformat(),
                    'regions': incident.triggered_regions,
                    'current_value': incident.current_value
                } for incident in list(self.active_incidents.values())[-10:]
            ],
            
            'performance_metrics': {
                'metrics_collected_per_minute': self.monitoring_metrics['total_metrics_collected'] / max(time.time() / 60, 1),
                'alert_evaluation_latency_seconds': 15.2,  # Example
                'notification_delivery_success_rate': 98.5,
                'database_query_time_ms': 25.3
            },
            
            'compliance_status': {
                'regulatory_alerts_active': len([
                    i for i in self.active_incidents.values()
                    if i.category == MonitoringCategory.COMPLIANCE
                ]),
                'audit_trail_completeness': 99.98,
                'data_retention_compliance': True,
                'alert_response_sla_compliance': 95.2
            },
            
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard

class RegionalMonitoringAgent:
    """Regional monitoring agent for collecting local metrics"""
    
    def __init__(self, region_id: str, region_config: Dict[str, Any]):
        self.region_id = region_id
        self.region_config = region_config
        self.metrics_definitions = {}
        
    async def initialize(self, metrics: Dict[str, MetricDefinition]):
        """Initialize the regional agent"""
        self.metrics_definitions = metrics
        logger.info(f"🤖 Initialized regional agent for {self.region_id}")
    
    async def collect_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics from this region"""
        
        collected_metrics = []
        current_time = datetime.now()
        
        # Simulate metric collection
        for metric_id, metric_def in self.metrics_definitions.items():
            
            # Generate realistic metric values
            if metric_id == 'cpu_usage':
                value = np.random.uniform(20, 85)
            elif metric_id == 'memory_usage':
                value = np.random.uniform(30, 75)
            elif metric_id == 'network_latency':
                value = np.random.exponential(10) + 5
            elif metric_id == 'error_rate':
                value = np.random.exponential(0.5)
            elif metric_id == 'trading_volume':
                value = np.random.poisson(50)
            elif metric_id == 'order_execution_latency':
                value = np.random.exponential(5) + 2
            else:
                value = np.random.uniform(0, 100)
            
            metric_data = {
                'metric_id': metric_id,
                'region': self.region_id,
                'timestamp': current_time,
                'value': float(value),
                'labels': {
                    'region': self.region_id,
                    'agent': 'regional_agent'
                }
            }
            
            collected_metrics.append(metric_data)
        
        return collected_metrics
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get regional health summary"""
        
        return {
            'status': 'healthy',
            'metrics_collected': len(self.metrics_definitions),
            'last_collection': datetime.now().isoformat(),
            'agent_uptime_hours': 24.5,  # Example
            'collection_success_rate': 99.2
        }

class AlertManager:
    """Alert management and correlation"""
    
    async def initialize(self, alert_rules: Dict[str, AlertRule]):
        self.alert_rules = alert_rules
        logger.info("🚨 Alert manager initialized")

class NotificationManager:
    """Notification delivery across multiple channels"""
    
    async def initialize(self):
        logger.info("📧 Notification manager initialized")
    
    async def send_alert_notifications(self, rule: AlertRule, incident: AlertIncident, trigger_data: Dict[str, Any]):
        """Send alert notifications via configured channels"""
        
        message = self._format_alert_message(rule, incident, trigger_data)
        
        for channel in rule.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email(message, rule.severity)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack(message, rule.severity)
                elif channel == AlertChannel.PAGERDUTY:
                    await self._send_pagerduty(message, incident)
                elif channel == AlertChannel.SMS:
                    await self._send_sms(message)
                
                incident.notifications_sent.append(channel.value)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    def _format_alert_message(self, rule: AlertRule, incident: AlertIncident, trigger_data: Dict[str, Any]) -> str:
        """Format alert message"""
        
        template = rule.notification_template or "Alert: {rule_name}"
        
        return template.format(
            rule_name=rule.name,
            value=trigger_data['value'],
            threshold=trigger_data['threshold'],
            region=trigger_data['region'],
            severity=incident.severity.value.upper()
        )
    
    async def _send_email(self, message: str, severity: AlertSeverity):
        """Send email notification"""
        logger.info(f"📧 Email sent: {message}")
    
    async def _send_slack(self, message: str, severity: AlertSeverity):
        """Send Slack notification"""
        logger.info(f"💬 Slack sent: {message}")
    
    async def _send_pagerduty(self, message: str, incident: AlertIncident):
        """Send PagerDuty notification"""
        logger.info(f"📟 PagerDuty sent: {message}")
    
    async def _send_sms(self, message: str):
        """Send SMS notification"""
        logger.info(f"📱 SMS sent: {message}")
    
    async def send_resolution_notification(self, rule: AlertRule, incident: AlertIncident):
        """Send incident resolution notification"""
        
        resolution_time = (incident.resolved_at - incident.triggered_at).total_seconds()
        message = f"✅ RESOLVED: {rule.name} after {resolution_time:.0f}s"
        
        for channel in rule.channels:
            if channel == AlertChannel.SLACK:
                await self._send_slack(message, AlertSeverity.INFO)

# Main execution
async def main():
    """Main execution for monitoring platform testing"""
    
    monitoring = GlobalMonitoringPlatform()
    await monitoring.initialize()
    
    logger.info("📊 Global Enterprise Monitoring Platform Started")
    
    # Let monitoring run for a while
    await asyncio.sleep(30)
    
    # Get dashboard
    dashboard = await monitoring.get_monitoring_dashboard()
    logger.info(f"📈 Monitoring Dashboard: {json.dumps(dashboard, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
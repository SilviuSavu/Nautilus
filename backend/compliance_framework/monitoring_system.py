"""
Real-Time Compliance Monitoring and Violation Detection System
=============================================================

Advanced monitoring system that provides real-time compliance oversight,
violation detection, and automated response capabilities.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from pathlib import Path
import uuid
import statistics
import aioredis
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests


class ViolationType(Enum):
    """Types of compliance violations"""
    DATA_RESIDENCY_VIOLATION = "data_residency_violation"
    CROSS_BORDER_TRANSFER_VIOLATION = "cross_border_transfer_violation"
    ACCESS_CONTROL_VIOLATION = "access_control_violation"
    RETENTION_POLICY_VIOLATION = "retention_policy_violation"
    ENCRYPTION_REQUIREMENT_VIOLATION = "encryption_requirement_violation"
    AUDIT_TRAIL_VIOLATION = "audit_trail_violation"
    PRIVACY_POLICY_VIOLATION = "privacy_policy_violation"
    CONSENT_MANAGEMENT_VIOLATION = "consent_management_violation"
    BREACH_NOTIFICATION_VIOLATION = "breach_notification_violation"
    REGULATORY_REPORTING_VIOLATION = "regulatory_reporting_violation"
    SYSTEM_SECURITY_VIOLATION = "system_security_violation"
    OPERATIONAL_RISK_VIOLATION = "operational_risk_violation"


class SeverityLevel(Enum):
    """Violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Monitoring system status"""
    ACTIVE = "active"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SYSLOG = "syslog"


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    violation_type: ViolationType
    severity: SeverityLevel
    title: str
    description: str
    affected_system: str
    affected_data: List[str]
    jurisdiction: str
    regulatory_framework: str
    detection_timestamp: datetime
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    status: str
    assigned_to: Optional[str]
    resolution_due: datetime
    resolution_timestamp: Optional[datetime]
    resolution_notes: Optional[str]
    evidence: Dict[str, Any]
    impact_assessment: Dict[str, Any]
    remediation_actions: List[Dict[str, Any]]
    escalation_level: int
    business_impact: str
    technical_details: Dict[str, Any]
    compliance_tags: Set[str]
    related_violations: List[str]
    cost_estimate: Optional[float]
    risk_score: float
    auto_remediation_attempted: bool
    notification_sent: bool
    regulatory_notification_required: bool
    external_reporting_status: str


@dataclass
class MonitoringRule:
    """Real-time monitoring rule definition"""
    rule_id: str
    name: str
    description: str
    violation_type: ViolationType
    severity: SeverityLevel
    condition: str
    data_sources: List[str]
    monitoring_frequency_seconds: int
    threshold_values: Dict[str, Any]
    evaluation_window_minutes: int
    aggregation_method: str
    baseline_values: Dict[str, Any]
    alert_channels: List[AlertChannel]
    auto_remediation_enabled: bool
    escalation_rules: List[Dict[str, Any]]
    business_hours_only: bool
    maintenance_windows: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_evaluation: Optional[datetime]
    evaluation_count: int
    violation_count: int
    false_positive_count: int
    is_active: bool


@dataclass
class SystemMetrics:
    """System performance and compliance metrics"""
    timestamp: datetime
    active_monitors: int
    total_evaluations_per_minute: float
    average_evaluation_time_ms: float
    violations_detected_last_hour: int
    violations_resolved_last_hour: int
    open_critical_violations: int
    open_high_violations: int
    compliance_score: float
    system_health_score: float
    data_quality_score: float
    alert_fatigue_score: float
    false_positive_rate: float
    resolution_time_p95: float
    uptime_percentage: float


class RealTimeComplianceMonitor:
    """
    Advanced real-time compliance monitoring system that continuously
    monitors for regulatory violations and triggers automated responses.
    """
    
    def __init__(self, 
                 data_directory: str = "/app/compliance/monitoring",
                 redis_url: str = "redis://localhost:6379"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Monitoring state
        self.status = MonitoringStatus.ACTIVE
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.active_violations: Dict[str, ComplianceViolation] = {}
        self.violation_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.evaluation_times: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=24*60)  # 24 hours of minute-level metrics
        
        # Alert management
        self.alert_channels: Dict[AlertChannel, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        
        self.logger = logging.getLogger("compliance.monitoring")
        
        # Initialize system
        asyncio.create_task(self._initialize_system())
    
    async def _initialize_system(self):
        """Initialize the monitoring system"""
        
        # Connect to Redis
        self.redis_client = aioredis.from_url(self.redis_url)
        
        # Load existing rules and violations
        self._load_monitoring_rules()
        self._load_active_violations()
        
        # Initialize default monitoring rules
        self._initialize_default_rules()
        
        # Configure alert channels
        self._configure_alert_channels()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._violation_cleanup_loop())
        asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Real-time compliance monitoring system initialized")
    
    def _initialize_default_rules(self):
        """Initialize comprehensive default monitoring rules"""
        
        # Data Residency Monitoring
        data_residency_rule = MonitoringRule(
            rule_id="DR-001",
            name="Data Residency Compliance Monitoring",
            description="Monitor for data stored outside of required jurisdictional boundaries",
            violation_type=ViolationType.DATA_RESIDENCY_VIOLATION,
            severity=SeverityLevel.CRITICAL,
            condition="data_location NOT IN allowed_locations",
            data_sources=["data_catalog", "storage_locations", "backup_systems"],
            monitoring_frequency_seconds=300,  # 5 minutes
            threshold_values={"max_violations": 0},
            evaluation_window_minutes=15,
            aggregation_method="count",
            baseline_values={"expected_violations": 0},
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            auto_remediation_enabled=True,
            escalation_rules=[
                {"level": 1, "delay_minutes": 15, "channels": ["email"]},
                {"level": 2, "delay_minutes": 60, "channels": ["slack", "pagerduty"]},
                {"level": 3, "delay_minutes": 240, "channels": ["webhook"]}
            ],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(data_residency_rule)
        
        # Cross-Border Transfer Monitoring
        transfer_rule = MonitoringRule(
            rule_id="CBT-001",
            name="Unauthorized Cross-Border Data Transfer Detection",
            description="Detect unauthorized cross-border data transfers without proper safeguards",
            violation_type=ViolationType.CROSS_BORDER_TRANSFER_VIOLATION,
            severity=SeverityLevel.HIGH,
            condition="cross_border_transfer = TRUE AND transfer_approval = NULL",
            data_sources=["data_transfers", "transfer_approvals", "network_logs"],
            monitoring_frequency_seconds=60,  # 1 minute
            threshold_values={"max_unauthorized_transfers": 0},
            evaluation_window_minutes=5,
            aggregation_method="count",
            baseline_values={"expected_transfers": 10},
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            auto_remediation_enabled=True,
            escalation_rules=[
                {"level": 1, "delay_minutes": 5, "channels": ["slack"]},
                {"level": 2, "delay_minutes": 30, "channels": ["email", "pagerduty"]}
            ],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(transfer_rule)
        
        # Access Control Monitoring
        access_rule = MonitoringRule(
            rule_id="AC-001",
            name="Unauthorized Access Detection",
            description="Monitor for unauthorized access to sensitive data",
            violation_type=ViolationType.ACCESS_CONTROL_VIOLATION,
            severity=SeverityLevel.HIGH,
            condition="failed_access_attempts > 5 OR admin_access_without_mfa = TRUE",
            data_sources=["audit_logs", "access_logs", "authentication_logs"],
            monitoring_frequency_seconds=120,  # 2 minutes
            threshold_values={"max_failed_attempts": 5, "admin_mfa_required": True},
            evaluation_window_minutes=10,
            aggregation_method="sum",
            baseline_values={"normal_failed_attempts": 2},
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SYSLOG],
            auto_remediation_enabled=False,
            escalation_rules=[
                {"level": 1, "delay_minutes": 0, "channels": ["syslog"]},
                {"level": 2, "delay_minutes": 10, "channels": ["slack"]},
                {"level": 3, "delay_minutes": 30, "channels": ["email"]}
            ],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(access_rule)
        
        # Data Retention Monitoring
        retention_rule = MonitoringRule(
            rule_id="DR-002",
            name="Data Retention Policy Compliance",
            description="Monitor for data retained beyond policy limits or deleted prematurely",
            violation_type=ViolationType.RETENTION_POLICY_VIOLATION,
            severity=SeverityLevel.MEDIUM,
            condition="retention_days > max_retention OR (deletion_required = TRUE AND data_exists = TRUE)",
            data_sources=["data_inventory", "retention_policies", "deletion_logs"],
            monitoring_frequency_seconds=3600,  # 1 hour
            threshold_values={"max_retention_exceedance": 30},
            evaluation_window_minutes=60,
            aggregation_method="count",
            baseline_values={"expected_retentions": 0},
            alert_channels=[AlertChannel.EMAIL],
            auto_remediation_enabled=True,
            escalation_rules=[
                {"level": 1, "delay_minutes": 60, "channels": ["email"]},
                {"level": 2, "delay_minutes": 480, "channels": ["slack"]}
            ],
            business_hours_only=True,
            maintenance_windows=[
                {"day": "sunday", "start_hour": 2, "end_hour": 6}
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(retention_rule)
        
        # Encryption Compliance Monitoring
        encryption_rule = MonitoringRule(
            rule_id="EC-001",
            name="Encryption Requirement Compliance",
            description="Monitor for sensitive data stored without required encryption",
            violation_type=ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION,
            severity=SeverityLevel.HIGH,
            condition="data_classification IN ('confidential', 'restricted') AND encryption_level < required_level",
            data_sources=["data_inventory", "encryption_status", "storage_systems"],
            monitoring_frequency_seconds=1800,  # 30 minutes
            threshold_values={"min_encryption_level": "high"},
            evaluation_window_minutes=30,
            aggregation_method="count",
            baseline_values={"expected_unencrypted": 0},
            alert_channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            auto_remediation_enabled=True,
            escalation_rules=[
                {"level": 1, "delay_minutes": 30, "channels": ["slack"]},
                {"level": 2, "delay_minutes": 120, "channels": ["email"]}
            ],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(encryption_rule)
        
        # Breach Notification Monitoring
        breach_rule = MonitoringRule(
            rule_id="BN-001",
            name="Data Breach Notification Compliance",
            description="Monitor for data breaches requiring regulatory notification within 72 hours",
            violation_type=ViolationType.BREACH_NOTIFICATION_VIOLATION,
            severity=SeverityLevel.CRITICAL,
            condition="data_breach_detected = TRUE AND hours_since_breach > 72 AND notification_sent = FALSE",
            data_sources=["security_incidents", "breach_assessments", "notification_logs"],
            monitoring_frequency_seconds=300,  # 5 minutes
            threshold_values={"notification_deadline_hours": 72},
            evaluation_window_minutes=5,
            aggregation_method="exists",
            baseline_values={},
            alert_channels=[AlertChannel.EMAIL, AlertChannel.PAGERDUTY, AlertChannel.WEBHOOK],
            auto_remediation_enabled=False,
            escalation_rules=[
                {"level": 1, "delay_minutes": 0, "channels": ["pagerduty"]},
                {"level": 2, "delay_minutes": 15, "channels": ["webhook"]},
                {"level": 3, "delay_minutes": 60, "channels": ["email"]}
            ],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        self.add_monitoring_rule(breach_rule)
    
    def add_monitoring_rule(self, rule: MonitoringRule) -> bool:
        """Add a monitoring rule to the system"""
        try:
            self.monitoring_rules[rule.rule_id] = rule
            self._save_monitoring_rule(rule)
            self.logger.info(f"Added monitoring rule: {rule.name} ({rule.rule_id})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add monitoring rule {rule.rule_id}: {str(e)}")
            return False
    
    async def _monitoring_loop(self):
        """Main monitoring loop that evaluates rules continuously"""
        while True:
            try:
                if self.status != MonitoringStatus.ACTIVE:
                    await asyncio.sleep(60)
                    continue
                
                # Evaluate all active rules
                evaluation_tasks = []
                for rule in self.monitoring_rules.values():
                    if rule.is_active and self._should_evaluate_rule(rule):
                        task = asyncio.create_task(self._evaluate_rule(rule))
                        evaluation_tasks.append(task)
                
                # Execute evaluations concurrently
                if evaluation_tasks:
                    start_time = time.time()
                    results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                    evaluation_time = time.time() - start_time
                    
                    self.evaluation_times.append(evaluation_time)
                    
                    # Process results
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Rule evaluation failed: {str(result)}")
                        elif result:
                            # Violation detected
                            await self._handle_violation_detection(result)
                
                await asyncio.sleep(30)  # Base monitoring interval
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    def _should_evaluate_rule(self, rule: MonitoringRule) -> bool:
        """Determine if a rule should be evaluated based on frequency and schedule"""
        
        now = datetime.now(timezone.utc)
        
        # Check if enough time has passed since last evaluation
        if rule.last_evaluation:
            time_since_last = (now - rule.last_evaluation).total_seconds()
            if time_since_last < rule.monitoring_frequency_seconds:
                return False
        
        # Check business hours restriction
        if rule.business_hours_only:
            # Simplified business hours check (9 AM - 6 PM UTC)
            if not (9 <= now.hour < 18):
                return False
        
        # Check maintenance windows
        for window in rule.maintenance_windows:
            if self._is_in_maintenance_window(now, window):
                return False
        
        return True
    
    def _is_in_maintenance_window(self, timestamp: datetime, window: Dict[str, Any]) -> bool:
        """Check if timestamp falls within a maintenance window"""
        
        # Simplified maintenance window check
        if window.get("day"):
            weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            current_weekday = weekday_names[timestamp.weekday()]
            
            if current_weekday == window["day"].lower():
                start_hour = window.get("start_hour", 0)
                end_hour = window.get("end_hour", 24)
                
                if start_hour <= timestamp.hour < end_hour:
                    return True
        
        return False
    
    async def _evaluate_rule(self, rule: MonitoringRule) -> Optional[ComplianceViolation]:
        """Evaluate a specific monitoring rule"""
        
        evaluation_start = time.time()
        
        try:
            # Update evaluation tracking
            rule.last_evaluation = datetime.now(timezone.utc)
            rule.evaluation_count += 1
            
            # Collect data from sources
            rule_data = await self._collect_rule_data(rule)
            
            # Evaluate condition
            violation_detected = self._evaluate_condition(rule, rule_data)
            
            if violation_detected:
                # Create violation record
                violation = await self._create_violation(rule, rule_data)
                rule.violation_count += 1
                return violation
            
            return None
            
        except Exception as e:
            self.logger.error(f"Rule evaluation failed for {rule.rule_id}: {str(e)}")
            return None
        
        finally:
            evaluation_time = time.time() - evaluation_start
            self.logger.debug(f"Rule {rule.rule_id} evaluated in {evaluation_time:.3f}s")
    
    async def _collect_rule_data(self, rule: MonitoringRule) -> Dict[str, Any]:
        """Collect data from rule data sources"""
        
        collected_data = {}
        
        for source in rule.data_sources:
            try:
                source_data = await self._query_data_source(source, rule)
                collected_data[source] = source_data
            except Exception as e:
                self.logger.error(f"Failed to collect data from {source}: {str(e)}")
                collected_data[source] = {"error": str(e)}
        
        return collected_data
    
    async def _query_data_source(self, source: str, rule: MonitoringRule) -> Dict[str, Any]:
        """Query a specific data source"""
        
        # Mock data source queries - in production, these would connect to real systems
        
        if source == "data_catalog":
            return {
                "total_records": 15420,
                "records_by_location": {
                    "us_east": 8500,
                    "eu_west": 6200,
                    "asia_pacific": 720
                },
                "violations": [
                    {"record_id": "rec-001", "location": "china", "required_location": "eu_west"},
                    {"record_id": "rec-002", "location": "russia", "required_location": "us_east"}
                ]
            }
        
        elif source == "audit_logs":
            return {
                "total_events": 45230,
                "failed_logins": 23,
                "admin_access_events": 156,
                "mfa_failures": 3,
                "suspicious_events": [
                    {"timestamp": "2024-08-23T14:30:00Z", "user": "admin001", "event": "login_without_mfa"},
                    {"timestamp": "2024-08-23T14:31:00Z", "user": "user123", "event": "multiple_failed_logins"}
                ]
            }
        
        elif source == "data_transfers":
            return {
                "total_transfers": 1240,
                "cross_border_transfers": 45,
                "unauthorized_transfers": [
                    {"transfer_id": "txf-001", "source": "eu", "destination": "china", "approval": None}
                ]
            }
        
        elif source == "encryption_status":
            return {
                "total_records": 15420,
                "encrypted_records": 14890,
                "unencrypted_sensitive": [
                    {"record_id": "rec-003", "classification": "confidential", "encryption": "none"}
                ]
            }
        
        else:
            return {"error": f"Unknown data source: {source}"}
    
    def _evaluate_condition(self, rule: MonitoringRule, data: Dict[str, Any]) -> bool:
        """Evaluate rule condition against collected data"""
        
        try:
            # Simplified condition evaluation - in production would use proper expression engine
            
            if rule.violation_type == ViolationType.DATA_RESIDENCY_VIOLATION:
                violations = data.get("data_catalog", {}).get("violations", [])
                return len(violations) > rule.threshold_values.get("max_violations", 0)
            
            elif rule.violation_type == ViolationType.CROSS_BORDER_TRANSFER_VIOLATION:
                unauthorized = data.get("data_transfers", {}).get("unauthorized_transfers", [])
                return len(unauthorized) > rule.threshold_values.get("max_unauthorized_transfers", 0)
            
            elif rule.violation_type == ViolationType.ACCESS_CONTROL_VIOLATION:
                audit_data = data.get("audit_logs", {})
                failed_logins = audit_data.get("failed_logins", 0)
                mfa_failures = audit_data.get("mfa_failures", 0)
                
                return (failed_logins > rule.threshold_values.get("max_failed_attempts", 5) or 
                       mfa_failures > 0)
            
            elif rule.violation_type == ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION:
                unencrypted = data.get("encryption_status", {}).get("unencrypted_sensitive", [])
                return len(unencrypted) > 0
            
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Condition evaluation failed for {rule.rule_id}: {str(e)}")
            return False
    
    async def _create_violation(self, rule: MonitoringRule, data: Dict[str, Any]) -> ComplianceViolation:
        """Create a compliance violation record"""
        
        violation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Extract evidence from data
        evidence = self._extract_violation_evidence(rule, data)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(rule, evidence)
        
        # Determine resolution deadline
        resolution_hours = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 24,
            SeverityLevel.MEDIUM: 72,
            SeverityLevel.LOW: 168
        }
        resolution_due = now + timedelta(hours=resolution_hours.get(rule.severity, 72))
        
        violation = ComplianceViolation(
            violation_id=violation_id,
            violation_type=rule.violation_type,
            severity=rule.severity,
            title=f"{rule.name} - {rule.violation_type.value}",
            description=f"Violation detected by rule: {rule.description}",
            affected_system=self._determine_affected_system(rule, evidence),
            affected_data=self._extract_affected_data(rule, evidence),
            jurisdiction=self._determine_jurisdiction(rule, evidence),
            regulatory_framework=self._determine_framework(rule, evidence),
            detection_timestamp=now,
            first_occurrence=now,
            last_occurrence=now,
            occurrence_count=1,
            status="open",
            assigned_to=None,
            resolution_due=resolution_due,
            resolution_timestamp=None,
            resolution_notes=None,
            evidence=evidence,
            impact_assessment=self._assess_impact(rule, evidence),
            remediation_actions=[],
            escalation_level=0,
            business_impact=self._assess_business_impact(rule.severity),
            technical_details=data,
            compliance_tags=set(),
            related_violations=[],
            cost_estimate=None,
            risk_score=risk_score,
            auto_remediation_attempted=False,
            notification_sent=False,
            regulatory_notification_required=self._requires_regulatory_notification(rule.severity),
            external_reporting_status="pending"
        )
        
        return violation
    
    def _extract_violation_evidence(self, rule: MonitoringRule, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evidence from rule evaluation data"""
        
        evidence = {
            "rule_id": rule.rule_id,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_sources": list(data.keys()),
            "threshold_values": rule.threshold_values
        }
        
        # Extract specific evidence based on violation type
        if rule.violation_type == ViolationType.DATA_RESIDENCY_VIOLATION:
            violations = data.get("data_catalog", {}).get("violations", [])
            evidence["residency_violations"] = violations
            evidence["violation_count"] = len(violations)
        
        elif rule.violation_type == ViolationType.CROSS_BORDER_TRANSFER_VIOLATION:
            unauthorized = data.get("data_transfers", {}).get("unauthorized_transfers", [])
            evidence["unauthorized_transfers"] = unauthorized
            evidence["transfer_count"] = len(unauthorized)
        
        elif rule.violation_type == ViolationType.ACCESS_CONTROL_VIOLATION:
            audit_data = data.get("audit_logs", {})
            evidence["failed_logins"] = audit_data.get("failed_logins", 0)
            evidence["suspicious_events"] = audit_data.get("suspicious_events", [])
        
        return evidence
    
    def _calculate_risk_score(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> float:
        """Calculate risk score for violation"""
        
        base_scores = {
            SeverityLevel.CRITICAL: 90.0,
            SeverityLevel.HIGH: 70.0,
            SeverityLevel.MEDIUM: 50.0,
            SeverityLevel.LOW: 30.0
        }
        
        base_score = base_scores.get(rule.severity, 50.0)
        
        # Adjust based on evidence
        multiplier = 1.0
        
        if rule.violation_type == ViolationType.DATA_RESIDENCY_VIOLATION:
            violation_count = evidence.get("violation_count", 0)
            multiplier = min(2.0, 1.0 + (violation_count * 0.1))
        
        elif rule.violation_type == ViolationType.CROSS_BORDER_TRANSFER_VIOLATION:
            transfer_count = evidence.get("transfer_count", 0)
            multiplier = min(2.0, 1.0 + (transfer_count * 0.2))
        
        return min(100.0, base_score * multiplier)
    
    def _determine_affected_system(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> str:
        """Determine which system is affected by the violation"""
        
        system_mapping = {
            ViolationType.DATA_RESIDENCY_VIOLATION: "Data Storage System",
            ViolationType.CROSS_BORDER_TRANSFER_VIOLATION: "Data Transfer System",
            ViolationType.ACCESS_CONTROL_VIOLATION: "Authentication System",
            ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION: "Data Encryption System"
        }
        
        return system_mapping.get(rule.violation_type, "Unknown System")
    
    def _extract_affected_data(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> List[str]:
        """Extract list of affected data from evidence"""
        
        affected_data = []
        
        if rule.violation_type == ViolationType.DATA_RESIDENCY_VIOLATION:
            violations = evidence.get("residency_violations", [])
            affected_data = [v.get("record_id", "unknown") for v in violations]
        
        elif rule.violation_type == ViolationType.CROSS_BORDER_TRANSFER_VIOLATION:
            transfers = evidence.get("unauthorized_transfers", [])
            affected_data = [t.get("transfer_id", "unknown") for t in transfers]
        
        return affected_data
    
    def _determine_jurisdiction(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> str:
        """Determine applicable jurisdiction for violation"""
        
        # Simplified jurisdiction determination
        violation_mapping = {
            ViolationType.DATA_RESIDENCY_VIOLATION: "EU",
            ViolationType.CROSS_BORDER_TRANSFER_VIOLATION: "GLOBAL",
            ViolationType.ACCESS_CONTROL_VIOLATION: "US",
            ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION: "GLOBAL"
        }
        
        return violation_mapping.get(rule.violation_type, "GLOBAL")
    
    def _determine_framework(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> str:
        """Determine applicable compliance framework"""
        
        framework_mapping = {
            ViolationType.DATA_RESIDENCY_VIOLATION: "GDPR",
            ViolationType.CROSS_BORDER_TRANSFER_VIOLATION: "GDPR/CCPA",
            ViolationType.ACCESS_CONTROL_VIOLATION: "SOC 2",
            ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION: "ISO 27001"
        }
        
        return framework_mapping.get(rule.violation_type, "General Compliance")
    
    def _assess_impact(self, rule: MonitoringRule, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of the violation"""
        
        return {
            "data_subjects_affected": evidence.get("violation_count", 1) * 100,
            "potential_fine": self._calculate_potential_fine(rule.severity),
            "reputational_impact": rule.severity.value,
            "operational_impact": "medium",
            "compliance_impact": "high" if rule.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] else "medium"
        }
    
    def _calculate_potential_fine(self, severity: SeverityLevel) -> str:
        """Calculate potential regulatory fine"""
        
        fine_estimates = {
            SeverityLevel.CRITICAL: "$1M - $10M",
            SeverityLevel.HIGH: "$100K - $1M",
            SeverityLevel.MEDIUM: "$10K - $100K",
            SeverityLevel.LOW: "$1K - $10K"
        }
        
        return fine_estimates.get(severity, "Unknown")
    
    def _assess_business_impact(self, severity: SeverityLevel) -> str:
        """Assess business impact of violation"""
        
        impact_mapping = {
            SeverityLevel.CRITICAL: "Severe - Operations at risk",
            SeverityLevel.HIGH: "High - Significant business disruption",
            SeverityLevel.MEDIUM: "Medium - Moderate business impact",
            SeverityLevel.LOW: "Low - Minimal business impact"
        }
        
        return impact_mapping.get(severity, "Unknown Impact")
    
    def _requires_regulatory_notification(self, severity: SeverityLevel) -> bool:
        """Determine if violation requires regulatory notification"""
        return severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    
    async def _handle_violation_detection(self, violation: ComplianceViolation):
        """Handle a newly detected violation"""
        
        # Check for existing similar violations
        existing_violation = self._find_existing_violation(violation)
        
        if existing_violation:
            # Update existing violation
            existing_violation.last_occurrence = violation.detection_timestamp
            existing_violation.occurrence_count += 1
            existing_violation.evidence.update(violation.evidence)
            
            self.logger.info(f"Updated existing violation: {existing_violation.violation_id}")
        else:
            # Add new violation
            self.active_violations[violation.violation_id] = violation
            self.violation_history.append(violation)
            
            self.logger.warning(f"New compliance violation detected: {violation.violation_id}")
            self.logger.warning(f"Type: {violation.violation_type.value}")
            self.logger.warning(f"Severity: {violation.severity.value}")
            self.logger.warning(f"Description: {violation.description}")
            
            # Send alerts
            await self._send_violation_alerts(violation)
            
            # Attempt auto-remediation if enabled
            if self._should_attempt_auto_remediation(violation):
                await self._attempt_auto_remediation(violation)
    
    def _find_existing_violation(self, violation: ComplianceViolation) -> Optional[ComplianceViolation]:
        """Find existing similar violation"""
        
        for existing in self.active_violations.values():
            if (existing.violation_type == violation.violation_type and
                existing.affected_system == violation.affected_system and
                existing.status == "open"):
                return existing
        
        return None
    
    async def _send_violation_alerts(self, violation: ComplianceViolation):
        """Send alerts for violation through configured channels"""
        
        # Find applicable monitoring rule
        rule = None
        for r in self.monitoring_rules.values():
            if r.violation_type == violation.violation_type:
                rule = r
                break
        
        if not rule:
            return
        
        # Send alerts through configured channels
        for channel in rule.alert_channels:
            try:
                await self._send_alert(channel, violation, rule)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.value}: {str(e)}")
        
        violation.notification_sent = True
    
    async def _send_alert(self, channel: AlertChannel, 
                         violation: ComplianceViolation, 
                         rule: MonitoringRule):
        """Send alert through specific channel"""
        
        # Check rate limits
        if not self._check_rate_limit(channel, violation.violation_type):
            self.logger.info(f"Rate limit exceeded for {channel.value}")
            return
        
        if channel == AlertChannel.EMAIL:
            await self._send_email_alert(violation, rule)
        elif channel == AlertChannel.SLACK:
            await self._send_slack_alert(violation, rule)
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook_alert(violation, rule)
        elif channel == AlertChannel.SYSLOG:
            await self._send_syslog_alert(violation, rule)
        
        # Update rate limit tracking
        self._update_rate_limit(channel, violation.violation_type)
    
    def _check_rate_limit(self, channel: AlertChannel, violation_type: ViolationType) -> bool:
        """Check if alert sending is within rate limits"""
        
        key = f"{channel.value}_{violation_type.value}"
        now = datetime.now(timezone.utc)
        
        # Clean old entries (older than 1 hour)
        self.rate_limits[key] = [
            ts for ts in self.rate_limits[key] 
            if (now - ts).total_seconds() < 3600
        ]
        
        # Check limit (max 5 alerts per hour per channel/type combination)
        return len(self.rate_limits[key]) < 5
    
    def _update_rate_limit(self, channel: AlertChannel, violation_type: ViolationType):
        """Update rate limit tracking"""
        key = f"{channel.value}_{violation_type.value}"
        self.rate_limits[key].append(datetime.now(timezone.utc))
    
    async def _send_email_alert(self, violation: ComplianceViolation, rule: MonitoringRule):
        """Send email alert (mock implementation)"""
        self.logger.info(f"EMAIL ALERT: {violation.title}")
        # In production, would send actual email
    
    async def _send_slack_alert(self, violation: ComplianceViolation, rule: MonitoringRule):
        """Send Slack alert (mock implementation)"""
        self.logger.info(f"SLACK ALERT: {violation.title}")
        # In production, would send to Slack webhook
    
    async def _send_webhook_alert(self, violation: ComplianceViolation, rule: MonitoringRule):
        """Send webhook alert (mock implementation)"""
        self.logger.info(f"WEBHOOK ALERT: {violation.title}")
        # In production, would POST to webhook URL
    
    async def _send_syslog_alert(self, violation: ComplianceViolation, rule: MonitoringRule):
        """Send syslog alert (mock implementation)"""
        self.logger.info(f"SYSLOG ALERT: {violation.title}")
        # In production, would send to syslog server
    
    def _should_attempt_auto_remediation(self, violation: ComplianceViolation) -> bool:
        """Determine if auto-remediation should be attempted"""
        
        # Find rule
        rule = None
        for r in self.monitoring_rules.values():
            if r.violation_type == violation.violation_type:
                rule = r
                break
        
        if not rule or not rule.auto_remediation_enabled:
            return False
        
        # Only attempt for certain violation types
        auto_remediation_types = [
            ViolationType.DATA_RESIDENCY_VIOLATION,
            ViolationType.RETENTION_POLICY_VIOLATION,
            ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION
        ]
        
        return violation.violation_type in auto_remediation_types
    
    async def _attempt_auto_remediation(self, violation: ComplianceViolation):
        """Attempt automatic remediation of violation"""
        
        self.logger.info(f"Attempting auto-remediation for violation: {violation.violation_id}")
        
        violation.auto_remediation_attempted = True
        
        try:
            if violation.violation_type == ViolationType.DATA_RESIDENCY_VIOLATION:
                await self._remediate_data_residency_violation(violation)
            elif violation.violation_type == ViolationType.ENCRYPTION_REQUIREMENT_VIOLATION:
                await self._remediate_encryption_violation(violation)
            elif violation.violation_type == ViolationType.RETENTION_POLICY_VIOLATION:
                await self._remediate_retention_violation(violation)
            
            self.logger.info(f"Auto-remediation completed for: {violation.violation_id}")
            
        except Exception as e:
            self.logger.error(f"Auto-remediation failed for {violation.violation_id}: {str(e)}")
    
    async def _remediate_data_residency_violation(self, violation: ComplianceViolation):
        """Auto-remediate data residency violations"""
        
        # Mock remediation - in production would:
        # 1. Identify affected data
        # 2. Move data to compliant location
        # 3. Update data catalog
        # 4. Verify remediation
        
        remediation_actions = [
            {"action": "identify_affected_data", "status": "completed"},
            {"action": "move_to_compliant_location", "status": "completed"},
            {"action": "update_catalog", "status": "completed"},
            {"action": "verify_compliance", "status": "completed"}
        ]
        
        violation.remediation_actions = remediation_actions
        violation.status = "auto_resolved"
        violation.resolution_timestamp = datetime.now(timezone.utc)
        violation.resolution_notes = "Auto-remediated by moving data to compliant location"
        
        self.logger.info(f"Data residency violation auto-remediated: {violation.violation_id}")
    
    async def _remediate_encryption_violation(self, violation: ComplianceViolation):
        """Auto-remediate encryption violations"""
        
        # Mock encryption remediation
        remediation_actions = [
            {"action": "identify_unencrypted_data", "status": "completed"},
            {"action": "apply_encryption", "status": "completed"},
            {"action": "verify_encryption", "status": "completed"}
        ]
        
        violation.remediation_actions = remediation_actions
        violation.status = "auto_resolved"
        violation.resolution_timestamp = datetime.now(timezone.utc)
        violation.resolution_notes = "Auto-remediated by applying required encryption"
        
        self.logger.info(f"Encryption violation auto-remediated: {violation.violation_id}")
    
    async def _remediate_retention_violation(self, violation: ComplianceViolation):
        """Auto-remediate retention policy violations"""
        
        # Mock retention remediation
        remediation_actions = [
            {"action": "identify_expired_data", "status": "completed"},
            {"action": "execute_deletion", "status": "completed"},
            {"action": "update_retention_records", "status": "completed"}
        ]
        
        violation.remediation_actions = remediation_actions
        violation.status = "auto_resolved"
        violation.resolution_timestamp = datetime.now(timezone.utc)
        violation.resolution_notes = "Auto-remediated by deleting expired data"
        
        self.logger.info(f"Retention violation auto-remediated: {violation.violation_id}")
    
    async def _metrics_collection_loop(self):
        """Collect and store system metrics"""
        while True:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Store metrics in Redis for external monitoring
                if self.redis_client:
                    metrics_data = asdict(metrics)
                    # Convert datetime to ISO string
                    metrics_data["timestamp"] = metrics.timestamp.isoformat()
                    
                    await self.redis_client.setex(
                        "compliance:metrics:current",
                        300,  # 5 minute expiry
                        json.dumps(metrics_data, default=str)
                    )
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(60)
    
    def _collect_current_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        
        now = datetime.now(timezone.utc)
        
        # Calculate evaluations per minute
        recent_evaluations = sum(
            1 for rule in self.monitoring_rules.values()
            if rule.last_evaluation and (now - rule.last_evaluation).total_seconds() < 60
        )
        
        # Calculate average evaluation time
        avg_eval_time = (
            statistics.mean(self.evaluation_times) * 1000 
            if self.evaluation_times else 0.0
        )
        
        # Count violations in last hour
        hour_ago = now - timedelta(hours=1)
        violations_last_hour = sum(
            1 for v in self.violation_history
            if v.detection_timestamp > hour_ago
        )
        
        resolved_last_hour = sum(
            1 for v in self.violation_history
            if (v.resolution_timestamp and v.resolution_timestamp > hour_ago)
        )
        
        # Count open violations by severity
        open_critical = sum(
            1 for v in self.active_violations.values()
            if v.status == "open" and v.severity == SeverityLevel.CRITICAL
        )
        
        open_high = sum(
            1 for v in self.active_violations.values()
            if v.status == "open" and v.severity == SeverityLevel.HIGH
        )
        
        # Calculate compliance score
        total_violations = len(self.active_violations)
        compliance_score = max(0, 100 - (total_violations * 5))  # Simplified calculation
        
        return SystemMetrics(
            timestamp=now,
            active_monitors=len([r for r in self.monitoring_rules.values() if r.is_active]),
            total_evaluations_per_minute=float(recent_evaluations),
            average_evaluation_time_ms=avg_eval_time,
            violations_detected_last_hour=violations_last_hour,
            violations_resolved_last_hour=resolved_last_hour,
            open_critical_violations=open_critical,
            open_high_violations=open_high,
            compliance_score=compliance_score,
            system_health_score=95.0,  # Mock value
            data_quality_score=92.0,   # Mock value
            alert_fatigue_score=15.0,  # Mock value
            false_positive_rate=2.5,   # Mock value
            resolution_time_p95=4.2,   # Mock value in hours
            uptime_percentage=99.9      # Mock value
        )
    
    async def _violation_cleanup_loop(self):
        """Cleanup resolved and expired violations"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                
                # Clean up resolved violations older than 30 days
                thirty_days_ago = now - timedelta(days=30)
                
                resolved_violations = [
                    v for v in self.active_violations.values()
                    if (v.status in ["resolved", "auto_resolved"] and
                        v.resolution_timestamp and
                        v.resolution_timestamp < thirty_days_ago)
                ]
                
                for violation in resolved_violations:
                    del self.active_violations[violation.violation_id]
                    self.logger.info(f"Cleaned up resolved violation: {violation.violation_id}")
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Error in violation cleanup: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _health_check_loop(self):
        """Monitor system health and update status"""
        while True:
            try:
                # Check Redis connectivity
                if self.redis_client:
                    await self.redis_client.ping()
                
                # Check rule evaluation health
                now = datetime.now(timezone.utc)
                stale_rules = [
                    rule for rule in self.monitoring_rules.values()
                    if (rule.is_active and rule.last_evaluation and
                        (now - rule.last_evaluation).total_seconds() > rule.monitoring_frequency_seconds * 3)
                ]
                
                if stale_rules:
                    self.logger.warning(f"{len(stale_rules)} monitoring rules appear stale")
                
                # Update system status
                if len(stale_rules) > len(self.monitoring_rules) * 0.5:
                    self.status = MonitoringStatus.ERROR
                elif len(stale_rules) > 0:
                    self.status = MonitoringStatus.PAUSED
                else:
                    self.status = MonitoringStatus.ACTIVE
                
                await asyncio.sleep(300)  # Health check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                self.status = MonitoringStatus.ERROR
                await asyncio.sleep(300)
    
    def _configure_alert_channels(self):
        """Configure alert notification channels"""
        
        self.alert_channels = {
            AlertChannel.EMAIL: {
                "smtp_server": "smtp.company.com",
                "port": 587,
                "username": "compliance@nautilus.com",
                "recipients": ["compliance@nautilus.com", "security@nautilus.com"]
            },
            AlertChannel.SLACK: {
                "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
                "channel": "#compliance-alerts"
            },
            AlertChannel.WEBHOOK: {
                "url": "https://api.nautilus.com/compliance/webhooks/violations",
                "auth_token": "webhook_auth_token_here"
            },
            AlertChannel.SYSLOG: {
                "server": "syslog.company.com",
                "port": 514,
                "facility": "local0"
            }
        }
    
    def _load_monitoring_rules(self):
        """Load monitoring rules from storage"""
        try:
            rule_files = self.data_directory.glob("rule_*.json")
            for rule_file in rule_files:
                with open(rule_file, 'r') as f:
                    rule_data = json.load(f)
                
                # Convert back to objects
                rule_data["violation_type"] = ViolationType(rule_data["violation_type"])
                rule_data["severity"] = SeverityLevel(rule_data["severity"])
                rule_data["alert_channels"] = [AlertChannel(ch) for ch in rule_data["alert_channels"]]
                rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
                rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
                
                if rule_data.get("last_evaluation"):
                    rule_data["last_evaluation"] = datetime.fromisoformat(rule_data["last_evaluation"])
                
                rule = MonitoringRule(**rule_data)
                self.monitoring_rules[rule.rule_id] = rule
            
            self.logger.info(f"Loaded {len(self.monitoring_rules)} monitoring rules")
            
        except Exception as e:
            self.logger.error(f"Error loading monitoring rules: {str(e)}")
    
    def _load_active_violations(self):
        """Load active violations from storage"""
        try:
            violation_files = self.data_directory.glob("violation_*.json")
            for violation_file in violation_files:
                with open(violation_file, 'r') as f:
                    violation_data = json.load(f)
                
                # Convert back to objects
                violation_data["violation_type"] = ViolationType(violation_data["violation_type"])
                violation_data["severity"] = SeverityLevel(violation_data["severity"])
                violation_data["detection_timestamp"] = datetime.fromisoformat(violation_data["detection_timestamp"])
                violation_data["first_occurrence"] = datetime.fromisoformat(violation_data["first_occurrence"])
                violation_data["last_occurrence"] = datetime.fromisoformat(violation_data["last_occurrence"])
                violation_data["resolution_due"] = datetime.fromisoformat(violation_data["resolution_due"])
                
                if violation_data.get("resolution_timestamp"):
                    violation_data["resolution_timestamp"] = datetime.fromisoformat(violation_data["resolution_timestamp"])
                
                violation_data["compliance_tags"] = set(violation_data["compliance_tags"])
                
                violation = ComplianceViolation(**violation_data)
                
                if violation.status == "open":
                    self.active_violations[violation.violation_id] = violation
                    self.violation_history.append(violation)
            
            self.logger.info(f"Loaded {len(self.active_violations)} active violations")
            
        except Exception as e:
            self.logger.error(f"Error loading violations: {str(e)}")
    
    def _save_monitoring_rule(self, rule: MonitoringRule):
        """Save monitoring rule to storage"""
        rule_file = self.data_directory / f"rule_{rule.rule_id}.json"
        
        rule_dict = asdict(rule)
        
        # Convert enums and datetime objects
        rule_dict["violation_type"] = rule.violation_type.value
        rule_dict["severity"] = rule.severity.value
        rule_dict["alert_channels"] = [ch.value for ch in rule.alert_channels]
        rule_dict["created_at"] = rule.created_at.isoformat()
        rule_dict["updated_at"] = rule.updated_at.isoformat()
        
        if rule.last_evaluation:
            rule_dict["last_evaluation"] = rule.last_evaluation.isoformat()
        
        with open(rule_file, 'w') as f:
            json.dump(rule_dict, f, indent=2, default=str)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        return {
            "status": self.status.value,
            "active_rules": len([r for r in self.monitoring_rules.values() if r.is_active]),
            "total_rules": len(self.monitoring_rules),
            "open_violations": len(self.active_violations),
            "critical_violations": len([v for v in self.active_violations.values() if v.severity == SeverityLevel.CRITICAL]),
            "last_evaluation": max((r.last_evaluation for r in self.monitoring_rules.values() if r.last_evaluation), default=None),
            "system_uptime": "99.9%",  # Mock value
            "average_evaluation_time_ms": statistics.mean(self.evaluation_times) * 1000 if self.evaluation_times else 0
        }
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of compliance violations"""
        
        violations_by_type = defaultdict(int)
        violations_by_severity = defaultdict(int)
        violations_by_status = defaultdict(int)
        
        for violation in self.active_violations.values():
            violations_by_type[violation.violation_type.value] += 1
            violations_by_severity[violation.severity.value] += 1
            violations_by_status[violation.status] += 1
        
        return {
            "total_active_violations": len(self.active_violations),
            "violations_by_type": dict(violations_by_type),
            "violations_by_severity": dict(violations_by_severity),
            "violations_by_status": dict(violations_by_status),
            "recent_violations": len([
                v for v in self.violation_history
                if (datetime.now(timezone.utc) - v.detection_timestamp).total_seconds() < 3600
            ])
        }
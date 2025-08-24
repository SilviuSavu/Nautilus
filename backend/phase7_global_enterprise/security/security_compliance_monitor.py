#!/usr/bin/env python3
"""
Phase 7: Security Compliance Monitor
Comprehensive compliance monitoring with regulatory frameworks and automated reporting
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import uuid
import re
from pathlib import Path
import pandas as pd
import numpy as np
from jinja2 import Template
import asyncpg
import redis.asyncio as redis
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"                         # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"                # Payment Card Industry DSS
    GDPR = "gdpr"                      # General Data Protection Regulation
    HIPAA = "hipaa"                    # Health Insurance Portability Act
    SOC2 = "soc2"                      # Service Organization Control 2
    ISO_27001 = "iso_27001"            # ISO/IEC 27001
    NIST_CSF = "nist_csf"              # NIST Cybersecurity Framework
    CCPA = "ccpa"                      # California Consumer Privacy Act
    FISMA = "fisma"                    # Federal Information Security Management Act
    BASEL_III = "basel_iii"            # Basel III banking regulation
    MiFID_II = "mifid_ii"              # Markets in Financial Instruments Directive
    COSO = "coso"                      # Committee of Sponsoring Organizations

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"            # Fully compliant
    PARTIALLY_COMPLIANT = "partially_compliant"  # Some issues
    NON_COMPLIANT = "non_compliant"    # Major violations
    NOT_APPLICABLE = "not_applicable"  # Framework doesn't apply
    UNDER_REVIEW = "under_review"      # Currently being assessed

class ViolationSeverity(Enum):
    """Compliance violation severity"""
    CRITICAL = "critical"              # Immediate action required
    HIGH = "high"                      # Action required within 24 hours
    MEDIUM = "medium"                  # Action required within 1 week
    LOW = "low"                        # Action required within 1 month
    INFORMATIONAL = "informational"    # No immediate action required

class AuditType(Enum):
    """Types of audits"""
    INTERNAL = "internal"              # Internal audit
    EXTERNAL = "external"              # External auditor
    REGULATORY = "regulatory"          # Regulatory body
    THIRD_PARTY = "third_party"        # Third-party assessment
    SELF_ASSESSMENT = "self_assessment" # Self-assessment

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    framework: ComplianceFramework
    category: str
    title: str
    description: str
    
    # Implementation details
    control_objective: str
    implementation_guidance: str
    evidence_requirements: List[str] = field(default_factory=list)
    
    # Assessment
    assessment_procedure: str = ""
    testing_frequency: str = "annual"  # annual, quarterly, monthly, continuous
    
    # Status
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW
    compliance_score: float = 0.0  # 0-100
    last_assessed: Optional[datetime] = None
    next_assessment_due: Optional[datetime] = None
    
    # Risk
    risk_level: ViolationSeverity = ViolationSeverity.MEDIUM
    business_impact: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    requirement_id: str
    framework: ComplianceFramework
    
    # Violation details
    title: str
    description: str
    severity: ViolationSeverity
    detected_at: datetime
    
    # Context
    affected_systems: List[str] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)
    root_cause: str = ""
    
    # Remediation
    remediation_plan: str = ""
    assigned_to: str = ""
    due_date: Optional[datetime] = None
    
    # Status
    status: str = "open"  # open, in_progress, resolved, closed
    resolution: str = ""
    resolved_at: Optional[datetime] = None
    
    # Evidence
    evidence_files: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    framework: ComplianceFramework
    report_type: str  # assessment, gap_analysis, remediation_status
    
    # Scope
    assessment_period_start: datetime
    assessment_period_end: datetime
    assessed_systems: List[str] = field(default_factory=list)
    
    # Results
    overall_compliance_score: float = 0.0
    compliant_requirements: int = 0
    non_compliant_requirements: int = 0
    partially_compliant_requirements: int = 0
    
    # Violations
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    remediation_priorities: List[str] = field(default_factory=list)
    
    # Metadata
    generated_by: str = "automated_system"
    generated_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

class SecurityComplianceMonitor:
    """
    Comprehensive security compliance monitoring system
    """
    
    def __init__(self):
        # Core components
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.reports: Dict[str, ComplianceReport] = {}
        
        # Framework mappings
        self.framework_requirements = {}
        self.control_mappings = {}
        
        # Data collectors
        self.data_collectors = {}
        
        # Evidence storage
        self.evidence_storage = {}
        
        # Database connections
        self.db_pool = None
        self.redis_client = None
        
        # Configuration
        self.config = {
            'assessment_schedule': {
                'continuous': ['access_controls', 'data_encryption'],
                'monthly': ['vulnerability_scans', 'patch_management'],
                'quarterly': ['risk_assessments', 'business_continuity'],
                'annual': ['penetration_testing', 'policy_reviews']
            },
            'alert_thresholds': {
                'critical_violations': 1,
                'compliance_score_threshold': 85.0,
                'overdue_remediation_days': 30
            },
            'report_formats': ['pdf', 'html', 'json', 'xml'],
            'retention_years': 7,
            'encryption_enabled': True
        }
        
        # Monitoring metrics
        self.compliance_metrics = {
            'total_requirements': 0,
            'compliant_requirements': 0,
            'non_compliant_requirements': 0,
            'active_violations': 0,
            'overdue_remediations': 0,
            'assessment_coverage': 0.0,
            'average_compliance_score': 0.0,
            'time_to_remediation_days': 0.0
        }
        
        # Active assessments
        self.active_assessments: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize compliance monitoring system"""
        logger.info("📋 Initializing Security Compliance Monitor")
        
        # Initialize database connections
        await self._initialize_databases()
        
        # Load compliance frameworks
        await self._load_compliance_frameworks()
        
        # Initialize data collectors
        await self._initialize_data_collectors()
        
        # Setup assessment schedules
        await self._setup_assessment_schedules()
        
        # Start monitoring loops
        await self._start_monitoring_loops()
        
        logger.info("✅ Security Compliance Monitor initialized")
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        
        # PostgreSQL for compliance data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://nautilus:password@postgres-compliance:5432/compliance",
            min_size=5,
            max_size=20
        )
        
        # Redis for caching and real-time data
        self.redis_client = redis.from_url(
            "redis://redis-compliance:6379",
            decode_responses=True
        )
        
        # Create compliance tables
        await self._create_compliance_tables()
        
        logger.info("✅ Compliance databases initialized")
    
    async def _create_compliance_tables(self):
        """Create compliance database tables"""
        
        async with self.db_pool.acquire() as conn:
            # Compliance requirements table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_requirements (
                    requirement_id VARCHAR PRIMARY KEY,
                    framework VARCHAR NOT NULL,
                    category VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    description TEXT NOT NULL,
                    control_objective TEXT NOT NULL,
                    implementation_guidance TEXT NOT NULL,
                    evidence_requirements TEXT[],
                    assessment_procedure TEXT,
                    testing_frequency VARCHAR DEFAULT 'annual',
                    status VARCHAR DEFAULT 'under_review',
                    compliance_score DOUBLE PRECISION DEFAULT 0.0,
                    last_assessed TIMESTAMPTZ,
                    next_assessment_due TIMESTAMPTZ,
                    risk_level VARCHAR DEFAULT 'medium',
                    business_impact TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Compliance violations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_violations (
                    violation_id VARCHAR PRIMARY KEY,
                    requirement_id VARCHAR NOT NULL REFERENCES compliance_requirements(requirement_id),
                    framework VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    description TEXT NOT NULL,
                    severity VARCHAR NOT NULL,
                    detected_at TIMESTAMPTZ NOT NULL,
                    affected_systems TEXT[],
                    affected_data TEXT[],
                    root_cause TEXT,
                    remediation_plan TEXT,
                    assigned_to VARCHAR,
                    due_date TIMESTAMPTZ,
                    status VARCHAR DEFAULT 'open',
                    resolution TEXT,
                    resolved_at TIMESTAMPTZ,
                    evidence_files TEXT[],
                    audit_trail JSONB DEFAULT '[]'
                )
            """)
            
            # Convert to hypertable for time-series optimization
            await conn.execute("""
                SELECT create_hypertable('compliance_violations', 'detected_at', if_not_exists => TRUE)
            """)
            
            # Compliance reports table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id VARCHAR PRIMARY KEY,
                    framework VARCHAR NOT NULL,
                    report_type VARCHAR NOT NULL,
                    assessment_period_start TIMESTAMPTZ NOT NULL,
                    assessment_period_end TIMESTAMPTZ NOT NULL,
                    assessed_systems TEXT[],
                    overall_compliance_score DOUBLE PRECISION,
                    compliant_requirements INTEGER DEFAULT 0,
                    non_compliant_requirements INTEGER DEFAULT 0,
                    partially_compliant_requirements INTEGER DEFAULT 0,
                    critical_violations INTEGER DEFAULT 0,
                    high_violations INTEGER DEFAULT 0,
                    medium_violations INTEGER DEFAULT 0,
                    low_violations INTEGER DEFAULT 0,
                    recommendations TEXT[],
                    remediation_priorities TEXT[],
                    generated_by VARCHAR DEFAULT 'automated_system',
                    generated_at TIMESTAMPTZ DEFAULT NOW(),
                    approved_by VARCHAR,
                    approved_at TIMESTAMPTZ
                )
            """)
            
            # Audit trail table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_audit_trail (
                    audit_id VARCHAR PRIMARY KEY,
                    entity_type VARCHAR NOT NULL, -- requirement, violation, report
                    entity_id VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    performed_by VARCHAR NOT NULL,
                    performed_at TIMESTAMPTZ DEFAULT NOW(),
                    old_values JSONB,
                    new_values JSONB,
                    reason TEXT,
                    ip_address VARCHAR,
                    user_agent VARCHAR
                )
            """)
            
            # Evidence storage table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_evidence (
                    evidence_id VARCHAR PRIMARY KEY,
                    requirement_id VARCHAR REFERENCES compliance_requirements(requirement_id),
                    violation_id VARCHAR REFERENCES compliance_violations(violation_id),
                    evidence_type VARCHAR NOT NULL, -- document, screenshot, log, config
                    file_name VARCHAR NOT NULL,
                    file_path VARCHAR NOT NULL,
                    file_hash VARCHAR NOT NULL,
                    file_size BIGINT NOT NULL,
                    mime_type VARCHAR,
                    encrypted BOOLEAN DEFAULT FALSE,
                    uploaded_by VARCHAR NOT NULL,
                    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
                    retention_until TIMESTAMPTZ
                )
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_requirements_framework ON compliance_requirements(framework)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_requirements_status ON compliance_requirements(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_severity ON compliance_violations(severity)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_status ON compliance_violations(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_trail_entity ON compliance_audit_trail(entity_type, entity_id)")
    
    async def _load_compliance_frameworks(self):
        """Load compliance frameworks and requirements"""
        
        logger.info("📚 Loading compliance frameworks")
        
        # SOX (Sarbanes-Oxley) requirements
        sox_requirements = [
            {
                'requirement_id': 'SOX-302',
                'framework': ComplianceFramework.SOX,
                'category': 'Management Certification',
                'title': 'CEO/CFO Certification',
                'description': 'Principal executive and financial officers must certify financial reports',
                'control_objective': 'Ensure accuracy and reliability of financial reporting',
                'implementation_guidance': 'Implement certification process for financial reports',
                'evidence_requirements': ['signed_certifications', 'supporting_documentation'],
                'risk_level': ViolationSeverity.CRITICAL
            },
            {
                'requirement_id': 'SOX-404',
                'framework': ComplianceFramework.SOX,
                'category': 'Internal Controls',
                'title': 'Management Assessment of Internal Controls',
                'description': 'Annual assessment of internal control over financial reporting',
                'control_objective': 'Maintain effective internal controls over financial reporting',
                'implementation_guidance': 'Document, test, and evaluate internal controls',
                'evidence_requirements': ['control_documentation', 'testing_results', 'deficiency_reports'],
                'risk_level': ViolationSeverity.CRITICAL
            }
        ]
        
        # PCI DSS requirements
        pci_requirements = [
            {
                'requirement_id': 'PCI-DSS-1',
                'framework': ComplianceFramework.PCI_DSS,
                'category': 'Network Security',
                'title': 'Install and maintain firewall configuration',
                'description': 'Build and maintain secure networks and systems',
                'control_objective': 'Protect cardholder data with secure network architecture',
                'implementation_guidance': 'Configure firewalls to restrict traffic between networks',
                'evidence_requirements': ['firewall_configs', 'network_diagrams', 'change_logs'],
                'testing_frequency': 'quarterly',
                'risk_level': ViolationSeverity.HIGH
            },
            {
                'requirement_id': 'PCI-DSS-3',
                'framework': ComplianceFramework.PCI_DSS,
                'category': 'Data Protection',
                'title': 'Protect stored cardholder data',
                'description': 'Protect stored cardholder data through encryption and secure storage',
                'control_objective': 'Minimize cardholder data storage and secure stored data',
                'implementation_guidance': 'Encrypt cardholder data and limit data retention',
                'evidence_requirements': ['encryption_configs', 'data_retention_policies', 'access_logs'],
                'testing_frequency': 'continuous',
                'risk_level': ViolationSeverity.CRITICAL
            }
        ]
        
        # GDPR requirements
        gdpr_requirements = [
            {
                'requirement_id': 'GDPR-ART-25',
                'framework': ComplianceFramework.GDPR,
                'category': 'Data Protection by Design',
                'title': 'Data protection by design and by default',
                'description': 'Implement technical and organizational measures for data protection',
                'control_objective': 'Ensure privacy protection is built into systems and processes',
                'implementation_guidance': 'Implement privacy-by-design principles',
                'evidence_requirements': ['privacy_impact_assessments', 'system_documentation', 'consent_records'],
                'testing_frequency': 'quarterly',
                'risk_level': ViolationSeverity.HIGH
            },
            {
                'requirement_id': 'GDPR-ART-32',
                'framework': ComplianceFramework.GDPR,
                'category': 'Security of Processing',
                'title': 'Security of processing',
                'description': 'Implement appropriate technical and organizational security measures',
                'control_objective': 'Ensure appropriate security of personal data',
                'implementation_guidance': 'Implement encryption, access controls, and security monitoring',
                'evidence_requirements': ['security_policies', 'encryption_evidence', 'access_logs', 'incident_reports'],
                'testing_frequency': 'monthly',
                'risk_level': ViolationSeverity.CRITICAL
            }
        ]
        
        # Combine all requirements
        all_requirements = sox_requirements + pci_requirements + gdpr_requirements
        
        # Store requirements
        for req_data in all_requirements:
            requirement = ComplianceRequirement(**req_data)
            self.requirements[requirement.requirement_id] = requirement
            
            # Store in database
            await self._store_requirement(requirement)
        
        logger.info(f"✅ Loaded {len(all_requirements)} compliance requirements")
    
    async def _store_requirement(self, requirement: ComplianceRequirement):
        """Store compliance requirement in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance_requirements (
                    requirement_id, framework, category, title, description,
                    control_objective, implementation_guidance, evidence_requirements,
                    assessment_procedure, testing_frequency, status, compliance_score,
                    last_assessed, next_assessment_due, risk_level, business_impact,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (requirement_id) DO UPDATE SET
                    framework = EXCLUDED.framework,
                    category = EXCLUDED.category,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    control_objective = EXCLUDED.control_objective,
                    implementation_guidance = EXCLUDED.implementation_guidance,
                    evidence_requirements = EXCLUDED.evidence_requirements,
                    assessment_procedure = EXCLUDED.assessment_procedure,
                    testing_frequency = EXCLUDED.testing_frequency,
                    risk_level = EXCLUDED.risk_level,
                    business_impact = EXCLUDED.business_impact,
                    updated_at = NOW()
            """,
            requirement.requirement_id,
            requirement.framework.value,
            requirement.category,
            requirement.title,
            requirement.description,
            requirement.control_objective,
            requirement.implementation_guidance,
            requirement.evidence_requirements,
            requirement.assessment_procedure,
            requirement.testing_frequency,
            requirement.status.value,
            requirement.compliance_score,
            requirement.last_assessed,
            requirement.next_assessment_due,
            requirement.risk_level.value,
            requirement.business_impact,
            requirement.created_at,
            requirement.updated_at
            )
    
    async def _initialize_data_collectors(self):
        """Initialize automated data collection systems"""
        
        # System configuration collector
        self.data_collectors['system_config'] = SystemConfigCollector()
        
        # Access log collector
        self.data_collectors['access_logs'] = AccessLogCollector()
        
        # Network security collector
        self.data_collectors['network_security'] = NetworkSecurityCollector()
        
        # Data encryption collector
        self.data_collectors['encryption'] = EncryptionCollector()
        
        logger.info("🔍 Data collectors initialized")
    
    async def _setup_assessment_schedules(self):
        """Setup automated assessment schedules"""
        
        for frequency, categories in self.config['assessment_schedule'].items():
            for category in categories:
                # Schedule assessments based on frequency
                await self._schedule_assessment(category, frequency)
        
        logger.info("📅 Assessment schedules configured")
    
    async def _start_monitoring_loops(self):
        """Start background monitoring tasks"""
        
        # Continuous compliance monitoring
        asyncio.create_task(self._continuous_monitoring_loop())
        
        # Scheduled assessments
        asyncio.create_task(self._scheduled_assessments_loop())
        
        # Violation detection and remediation tracking
        asyncio.create_task(self._violation_monitoring_loop())
        
        # Report generation
        asyncio.create_task(self._report_generation_loop())
        
        logger.info("🔄 Compliance monitoring loops started")
    
    async def _continuous_monitoring_loop(self):
        """Continuous compliance monitoring"""
        
        while True:
            try:
                # Check critical requirements continuously
                critical_requirements = [
                    req for req in self.requirements.values()
                    if req.testing_frequency == 'continuous'
                ]
                
                for requirement in critical_requirements:
                    assessment_result = await self._assess_requirement(requirement)
                    
                    if assessment_result['status'] == ComplianceStatus.NON_COMPLIANT:
                        await self._create_violation(requirement, assessment_result)
                    
                    await self._update_requirement_status(requirement, assessment_result)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
            
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _scheduled_assessments_loop(self):
        """Execute scheduled compliance assessments"""
        
        while True:
            try:
                # Check for due assessments
                due_assessments = await self._get_due_assessments()
                
                for requirement_id in due_assessments:
                    requirement = self.requirements.get(requirement_id)
                    if requirement:
                        await self._execute_assessment(requirement)
                
            except Exception as e:
                logger.error(f"Error in scheduled assessments: {e}")
            
            await asyncio.sleep(3600)  # Every hour
    
    async def _assess_requirement(self, requirement: ComplianceRequirement) -> Dict[str, Any]:
        """Assess compliance requirement"""
        
        assessment_result = {
            'requirement_id': requirement.requirement_id,
            'status': ComplianceStatus.COMPLIANT,
            'score': 100.0,
            'findings': [],
            'evidence': [],
            'recommendations': []
        }
        
        try:
            # Collect evidence based on requirement type
            if 'firewall' in requirement.title.lower():
                evidence = await self.data_collectors['network_security'].collect_firewall_config()
                assessment_result['evidence'].append(evidence)
                
                # Assess firewall configuration
                if not evidence.get('properly_configured', True):
                    assessment_result['status'] = ComplianceStatus.NON_COMPLIANT
                    assessment_result['score'] = 20.0
                    assessment_result['findings'].append('Firewall not properly configured')
            
            elif 'encryption' in requirement.title.lower():
                evidence = await self.data_collectors['encryption'].collect_encryption_status()
                assessment_result['evidence'].append(evidence)
                
                # Assess encryption implementation
                encryption_score = evidence.get('encryption_coverage', 0) * 100
                assessment_result['score'] = encryption_score
                
                if encryption_score < 95:
                    assessment_result['status'] = ComplianceStatus.PARTIALLY_COMPLIANT
                    assessment_result['findings'].append('Incomplete encryption coverage')
            
            elif 'access' in requirement.title.lower():
                evidence = await self.data_collectors['access_logs'].collect_access_controls()
                assessment_result['evidence'].append(evidence)
                
                # Assess access controls
                if evidence.get('unauthorized_access_attempts', 0) > 10:
                    assessment_result['status'] = ComplianceStatus.PARTIALLY_COMPLIANT
                    assessment_result['score'] = 75.0
                    assessment_result['findings'].append('High number of unauthorized access attempts')
            
            # Default assessment for other requirements
            else:
                # Simulate assessment based on requirement characteristics
                import random
                base_score = random.uniform(80, 100)
                
                if requirement.risk_level == ViolationSeverity.CRITICAL:
                    # Higher standards for critical requirements
                    if base_score < 95:
                        assessment_result['status'] = ComplianceStatus.PARTIALLY_COMPLIANT
                        assessment_result['score'] = base_score
                else:
                    assessment_result['score'] = base_score
                    if base_score < 85:
                        assessment_result['status'] = ComplianceStatus.PARTIALLY_COMPLIANT
        
        except Exception as e:
            logger.error(f"Error assessing requirement {requirement.requirement_id}: {e}")
            assessment_result['status'] = ComplianceStatus.NON_COMPLIANT
            assessment_result['score'] = 0.0
            assessment_result['findings'].append(f"Assessment failed: {str(e)}")
        
        return assessment_result
    
    async def _create_violation(self, requirement: ComplianceRequirement, assessment_result: Dict[str, Any]):
        """Create compliance violation record"""
        
        violation = ComplianceViolation(
            violation_id=str(uuid.uuid4()),
            requirement_id=requirement.requirement_id,
            framework=requirement.framework,
            title=f"Violation of {requirement.title}",
            description=f"Assessment found non-compliance: {'; '.join(assessment_result['findings'])}",
            severity=requirement.risk_level,
            detected_at=datetime.now(),
            root_cause="Assessment findings indicate non-compliance",
            remediation_plan="Review and remediate identified issues",
            status="open"
        )
        
        self.violations[violation.violation_id] = violation
        
        # Store in database
        await self._store_violation(violation)
        
        # Create audit trail entry
        await self._create_audit_entry(
            'violation',
            violation.violation_id,
            'created',
            'automated_assessment',
            new_values=asdict(violation)
        )
        
        logger.warning(f"🚨 Compliance violation created: {violation.title}")
    
    async def _store_violation(self, violation: ComplianceViolation):
        """Store compliance violation in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance_violations (
                    violation_id, requirement_id, framework, title, description,
                    severity, detected_at, affected_systems, affected_data,
                    root_cause, remediation_plan, assigned_to, due_date,
                    status, resolution, resolved_at, evidence_files, audit_trail
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            """,
            violation.violation_id,
            violation.requirement_id,
            violation.framework.value,
            violation.title,
            violation.description,
            violation.severity.value,
            violation.detected_at,
            violation.affected_systems,
            violation.affected_data,
            violation.root_cause,
            violation.remediation_plan,
            violation.assigned_to,
            violation.due_date,
            violation.status,
            violation.resolution,
            violation.resolved_at,
            violation.evidence_files,
            json.dumps([asdict(entry) for entry in violation.audit_trail], default=str)
            )
    
    async def generate_compliance_report(self, framework: ComplianceFramework, report_type: str = "assessment") -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        logger.info(f"📊 Generating {framework.value} compliance report")
        
        # Calculate assessment period (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get framework requirements
        framework_requirements = [
            req for req in self.requirements.values()
            if req.framework == framework
        ]
        
        # Calculate compliance metrics
        compliant = len([r for r in framework_requirements if r.status == ComplianceStatus.COMPLIANT])
        non_compliant = len([r for r in framework_requirements if r.status == ComplianceStatus.NON_COMPLIANT])
        partially_compliant = len([r for r in framework_requirements if r.status == ComplianceStatus.PARTIALLY_COMPLIANT])
        
        # Calculate overall compliance score
        total_score = sum(req.compliance_score for req in framework_requirements)
        overall_score = total_score / len(framework_requirements) if framework_requirements else 0
        
        # Get violations for this framework
        framework_violations = [
            v for v in self.violations.values()
            if v.framework == framework and start_date <= v.detected_at <= end_date
        ]
        
        # Count violations by severity
        critical_violations = len([v for v in framework_violations if v.severity == ViolationSeverity.CRITICAL])
        high_violations = len([v for v in framework_violations if v.severity == ViolationSeverity.HIGH])
        medium_violations = len([v for v in framework_violations if v.severity == ViolationSeverity.MEDIUM])
        low_violations = len([v for v in framework_violations if v.severity == ViolationSeverity.LOW])
        
        # Generate recommendations
        recommendations = []
        if non_compliant > 0:
            recommendations.append(f"Address {non_compliant} non-compliant requirements immediately")
        if critical_violations > 0:
            recommendations.append(f"Remediate {critical_violations} critical violations within 24 hours")
        if overall_score < 90:
            recommendations.append("Implement continuous monitoring for improved compliance")
        
        # Create report
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            framework=framework,
            report_type=report_type,
            assessment_period_start=start_date,
            assessment_period_end=end_date,
            assessed_systems=["trading_platform", "risk_management", "data_warehouse"],
            overall_compliance_score=overall_score,
            compliant_requirements=compliant,
            non_compliant_requirements=non_compliant,
            partially_compliant_requirements=partially_compliant,
            critical_violations=critical_violations,
            high_violations=high_violations,
            medium_violations=medium_violations,
            low_violations=low_violations,
            recommendations=recommendations,
            remediation_priorities=[
                "Critical security vulnerabilities",
                "Access control improvements",
                "Data encryption gaps",
                "Audit trail completeness"
            ]
        )
        
        self.reports[report.report_id] = report
        
        # Store in database
        await self._store_report(report)
        
        # Generate report files
        await self._generate_report_files(report)
        
        logger.info(f"✅ Compliance report generated: {report.report_id}")
        
        return report
    
    async def _store_report(self, report: ComplianceReport):
        """Store compliance report in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO compliance_reports (
                    report_id, framework, report_type, assessment_period_start,
                    assessment_period_end, assessed_systems, overall_compliance_score,
                    compliant_requirements, non_compliant_requirements,
                    partially_compliant_requirements, critical_violations,
                    high_violations, medium_violations, low_violations,
                    recommendations, remediation_priorities, generated_by,
                    generated_at, approved_by, approved_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """,
            report.report_id,
            report.framework.value,
            report.report_type,
            report.assessment_period_start,
            report.assessment_period_end,
            report.assessed_systems,
            report.overall_compliance_score,
            report.compliant_requirements,
            report.non_compliant_requirements,
            report.partially_compliant_requirements,
            report.critical_violations,
            report.high_violations,
            report.medium_violations,
            report.low_violations,
            report.recommendations,
            report.remediation_priorities,
            report.generated_by,
            report.generated_at,
            report.approved_by,
            report.approved_at
            )
    
    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard"""
        
        # Overall compliance status
        total_requirements = len(self.requirements)
        compliant_requirements = len([r for r in self.requirements.values() if r.status == ComplianceStatus.COMPLIANT])
        non_compliant_requirements = len([r for r in self.requirements.values() if r.status == ComplianceStatus.NON_COMPLIANT])
        
        overall_compliance_rate = (compliant_requirements / total_requirements * 100) if total_requirements > 0 else 0
        
        # Violations summary
        active_violations = len([v for v in self.violations.values() if v.status == 'open'])
        critical_violations = len([v for v in self.violations.values() if v.severity == ViolationSeverity.CRITICAL and v.status == 'open'])
        
        # Framework breakdown
        framework_status = {}
        for framework in ComplianceFramework:
            framework_reqs = [r for r in self.requirements.values() if r.framework == framework]
            if framework_reqs:
                compliant_count = len([r for r in framework_reqs if r.status == ComplianceStatus.COMPLIANT])
                framework_status[framework.value] = {
                    'total_requirements': len(framework_reqs),
                    'compliant_requirements': compliant_count,
                    'compliance_rate': (compliant_count / len(framework_reqs) * 100) if framework_reqs else 0,
                    'violations': len([v for v in self.violations.values() if v.framework == framework and v.status == 'open'])
                }
        
        # Recent activity
        recent_violations = sorted(
            [v for v in self.violations.values()],
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        dashboard = {
            'overview': {
                'total_requirements': total_requirements,
                'overall_compliance_rate': round(overall_compliance_rate, 2),
                'compliant_requirements': compliant_requirements,
                'non_compliant_requirements': non_compliant_requirements,
                'active_violations': active_violations,
                'critical_violations': critical_violations,
                'frameworks_monitored': len([f for f in framework_status if framework_status[f]['total_requirements'] > 0])
            },
            
            'framework_compliance': framework_status,
            
            'violation_summary': {
                'by_severity': {
                    'critical': len([v for v in self.violations.values() if v.severity == ViolationSeverity.CRITICAL and v.status == 'open']),
                    'high': len([v for v in self.violations.values() if v.severity == ViolationSeverity.HIGH and v.status == 'open']),
                    'medium': len([v for v in self.violations.values() if v.severity == ViolationSeverity.MEDIUM and v.status == 'open']),
                    'low': len([v for v in self.violations.values() if v.severity == ViolationSeverity.LOW and v.status == 'open'])
                },
                'by_status': {
                    'open': len([v for v in self.violations.values() if v.status == 'open']),
                    'in_progress': len([v for v in self.violations.values() if v.status == 'in_progress']),
                    'resolved': len([v for v in self.violations.values() if v.status == 'resolved'])
                }
            },
            
            'recent_violations': [
                {
                    'violation_id': v.violation_id,
                    'title': v.title,
                    'framework': v.framework.value,
                    'severity': v.severity.value,
                    'detected_at': v.detected_at.isoformat(),
                    'status': v.status
                } for v in recent_violations
            ],
            
            'assessment_coverage': {
                'continuous_monitoring': len([r for r in self.requirements.values() if r.testing_frequency == 'continuous']),
                'monthly_assessments': len([r for r in self.requirements.values() if r.testing_frequency == 'monthly']),
                'quarterly_assessments': len([r for r in self.requirements.values() if r.testing_frequency == 'quarterly']),
                'annual_assessments': len([r for r in self.requirements.values() if r.testing_frequency == 'annual'])
            },
            
            'remediation_status': {
                'overdue_remediations': len([v for v in self.violations.values() if v.due_date and v.due_date < datetime.now() and v.status != 'resolved']),
                'average_remediation_time_days': 15.5,  # Example
                'remediation_success_rate': 87.2
            },
            
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard

# Data collector classes
class SystemConfigCollector:
    """Collect system configuration data"""
    
    async def collect_system_config(self) -> Dict[str, Any]:
        return {
            'properly_configured': True,
            'security_settings': {'encryption_enabled': True, 'access_controls': True}
        }

class AccessLogCollector:
    """Collect access control data"""
    
    async def collect_access_controls(self) -> Dict[str, Any]:
        return {
            'unauthorized_access_attempts': 5,
            'successful_authentications': 1250,
            'failed_authentications': 23
        }

class NetworkSecurityCollector:
    """Collect network security data"""
    
    async def collect_firewall_config(self) -> Dict[str, Any]:
        return {
            'properly_configured': True,
            'rules_count': 157,
            'last_updated': datetime.now().isoformat()
        }

class EncryptionCollector:
    """Collect encryption status data"""
    
    async def collect_encryption_status(self) -> Dict[str, Any]:
        return {
            'encryption_coverage': 0.96,  # 96% coverage
            'encryption_algorithms': ['AES-256', 'RSA-2048'],
            'key_management': 'compliant'
        }

# Main execution
async def main():
    """Main execution for compliance monitoring testing"""
    
    monitor = SecurityComplianceMonitor()
    await monitor.initialize()
    
    logger.info("📋 Security Compliance Monitor started")
    
    # Generate sample reports
    sox_report = await monitor.generate_compliance_report(ComplianceFramework.SOX)
    pci_report = await monitor.generate_compliance_report(ComplianceFramework.PCI_DSS)
    gdpr_report = await monitor.generate_compliance_report(ComplianceFramework.GDPR)
    
    # Get dashboard
    dashboard = await monitor.get_compliance_dashboard()
    logger.info(f"📊 Compliance Dashboard: {json.dumps(dashboard, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
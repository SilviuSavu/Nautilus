"""
Multi-Region Compliance Policy Engine
====================================

Advanced policy engine that enforces regulatory requirements across multiple
jurisdictions including EU GDPR, US regulations, APAC requirements.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import uuid
from pydantic import BaseModel


class JurisdictionType(Enum):
    """Regulatory jurisdictions"""
    EU_GDPR = "eu_gdpr"
    US_FINRA = "us_finra"
    US_SEC = "us_sec"
    US_CFTC = "us_cftc"
    UK_FCA = "uk_fca"
    SINGAPORE_MAS = "singapore_mas"
    HONG_KONG_SFC = "hong_kong_sfc"
    JAPAN_FSA = "japan_fsa"
    AUSTRALIA_ASIC = "australia_asic"
    BASEL_III = "basel_iii"


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    SOC_2_TYPE_II = "soc_2_type_ii"
    ISO_27001 = "iso_27001"
    BASEL_III = "basel_iii"
    GDPR = "gdpr"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    FINRA_4511 = "finra_4511"
    SEC_RULE_15C3_3 = "sec_rule_15c3_3"
    MAS_AML_CFT = "mas_aml_cft"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class RiskLevel(Enum):
    """Compliance risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CompliancePolicy:
    """Compliance policy definition"""
    policy_id: str
    name: str
    jurisdiction: JurisdictionType
    framework: ComplianceFramework
    description: str
    requirements: List[str]
    controls: List[str]
    risk_level: RiskLevel
    data_classification: DataClassification
    retention_period_days: int
    cross_border_restrictions: List[str]
    automated_enforcement: bool
    monitoring_frequency_seconds: int
    violation_response: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    dependencies: List[str] = None
    exemptions: List[str] = None


class CompliancePolicyEngine:
    """
    Advanced policy engine for multi-region compliance enforcement.
    
    Manages regulatory policies across multiple jurisdictions and frameworks,
    provides real-time enforcement, and handles cross-border requirements.
    """
    
    def __init__(self, data_directory: str = "/app/compliance/policies"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        self.policies: Dict[str, CompliancePolicy] = {}
        self.jurisdiction_policies: Dict[JurisdictionType, List[str]] = {}
        self.framework_policies: Dict[ComplianceFramework, List[str]] = {}
        self.active_violations: Set[str] = set()
        
        self.logger = logging.getLogger("compliance.policy_engine")
        
        # Initialize with default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize with comprehensive default compliance policies"""
        
        # EU GDPR Data Protection
        gdpr_policy = CompliancePolicy(
            policy_id="GDPR-001",
            name="EU GDPR Data Protection and Privacy",
            jurisdiction=JurisdictionType.EU_GDPR,
            framework=ComplianceFramework.GDPR,
            description="Complete GDPR compliance for personal data processing",
            requirements=[
                "Lawful basis for processing personal data",
                "Data subject consent management",
                "Right to be forgotten implementation",
                "Data portability support",
                "Privacy by design and default",
                "Data Protection Officer appointment",
                "Data breach notification within 72 hours",
                "International transfer safeguards"
            ],
            controls=[
                "Pseudonymization and anonymization",
                "Access controls and authentication",
                "Audit logging of data access",
                "Regular security assessments",
                "Staff training and awareness",
                "Vendor due diligence",
                "Data inventory and mapping"
            ],
            risk_level=RiskLevel.CRITICAL,
            data_classification=DataClassification.RESTRICTED,
            retention_period_days=2555,  # 7 years
            cross_border_restrictions=[
                "No transfer to non-adequate countries without safeguards",
                "Standard Contractual Clauses required",
                "Binding Corporate Rules for intra-group transfers"
            ],
            automated_enforcement=True,
            monitoring_frequency_seconds=300,  # 5 minutes
            violation_response={
                "immediate_actions": ["alert_dpo", "log_incident", "assess_impact"],
                "escalation_levels": ["low", "medium", "high", "critical"],
                "notification_requirements": ["supervisory_authority", "data_subjects"],
                "remediation_timeline_hours": 72
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version="1.2.0",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        self.add_policy(gdpr_policy)
        
        # US FINRA Market Surveillance
        finra_policy = CompliancePolicy(
            policy_id="FINRA-001",
            name="FINRA Market Surveillance and Trade Reporting",
            jurisdiction=JurisdictionType.US_FINRA,
            framework=ComplianceFramework.FINRA_4511,
            description="FINRA compliance for market surveillance and trade reporting",
            requirements=[
                "Real-time market surveillance",
                "Order audit trail system (OATS)",
                "Trade reporting to FINRA",
                "Best execution documentation",
                "Anti-money laundering (AML) monitoring",
                "Customer identification program",
                "Suspicious activity reporting",
                "Books and records maintenance"
            ],
            controls=[
                "Automated surveillance systems",
                "Exception reporting and investigation",
                "Regular compliance testing",
                "Staff supervision and training",
                "Independent compliance function",
                "Risk assessment frameworks",
                "Vendor oversight programs"
            ],
            risk_level=RiskLevel.CRITICAL,
            data_classification=DataClassification.CONFIDENTIAL,
            retention_period_days=2190,  # 6 years
            cross_border_restrictions=[
                "US person data must remain in US jurisdiction",
                "Cross-border sharing requires regulatory approval"
            ],
            automated_enforcement=True,
            monitoring_frequency_seconds=60,  # 1 minute for trading surveillance
            violation_response={
                "immediate_actions": ["halt_trading", "preserve_records", "notify_finra"],
                "escalation_levels": ["minor", "material", "systemic"],
                "notification_requirements": ["finra", "sec", "internal_compliance"],
                "remediation_timeline_hours": 24
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version="1.1.0",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        self.add_policy(finra_policy)
        
        # Basel III Capital Requirements
        basel_policy = CompliancePolicy(
            policy_id="BASEL-001",
            name="Basel III Capital and Liquidity Requirements",
            jurisdiction=JurisdictionType.BASEL_III,
            framework=ComplianceFramework.BASEL_III,
            description="Basel III implementation for capital adequacy and risk management",
            requirements=[
                "Common Equity Tier 1 capital ratio >= 4.5%",
                "Tier 1 capital ratio >= 6%",
                "Total capital ratio >= 8%",
                "Leverage ratio >= 3%",
                "Liquidity Coverage Ratio >= 100%",
                "Net Stable Funding Ratio >= 100%",
                "Operational risk capital allocation",
                "Market risk capital requirements"
            ],
            controls=[
                "Daily capital calculations",
                "Real-time risk monitoring",
                "Stress testing scenarios",
                "Model validation programs",
                "Independent risk management",
                "Supervisory review processes",
                "Market discipline disclosures"
            ],
            risk_level=RiskLevel.CRITICAL,
            data_classification=DataClassification.CONFIDENTIAL,
            retention_period_days=3650,  # 10 years
            cross_border_restrictions=[
                "Cross-jurisdictional capital recognition",
                "Home-host supervisory coordination"
            ],
            automated_enforcement=True,
            monitoring_frequency_seconds=3600,  # 1 hour
            violation_response={
                "immediate_actions": ["capital_preservation", "risk_reduction", "supervisor_notify"],
                "escalation_levels": ["early_warning", "prompt_corrective", "critical"],
                "notification_requirements": ["national_supervisor", "home_supervisor"],
                "remediation_timeline_hours": 168  # 7 days
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version="1.0.0",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        self.add_policy(basel_policy)
        
        # Singapore MAS Technology Risk Management
        mas_policy = CompliancePolicy(
            policy_id="MAS-001",
            name="Singapore MAS Technology Risk Management",
            jurisdiction=JurisdictionType.SINGAPORE_MAS,
            framework=ComplianceFramework.MAS_AML_CFT,
            description="MAS guidelines for technology risk management and cybersecurity",
            requirements=[
                "Technology risk management framework",
                "Cybersecurity controls and monitoring",
                "Business continuity and disaster recovery",
                "Outsourcing risk management",
                "Data governance and protection",
                "System reliability and availability",
                "Incident management and reporting",
                "Regular security assessments"
            ],
            controls=[
                "Multi-factor authentication",
                "Network segmentation and monitoring",
                "Endpoint detection and response",
                "Security incident response team",
                "Regular penetration testing",
                "Vendor security assessments",
                "Staff security training"
            ],
            risk_level=RiskLevel.HIGH,
            data_classification=DataClassification.CONFIDENTIAL,
            retention_period_days=2190,  # 6 years
            cross_border_restrictions=[
                "Singapore customer data residency requirements",
                "Cross-border data transfer approval required"
            ],
            automated_enforcement=True,
            monitoring_frequency_seconds=300,  # 5 minutes
            violation_response={
                "immediate_actions": ["isolate_systems", "preserve_evidence", "notify_mas"],
                "escalation_levels": ["low", "medium", "high", "critical"],
                "notification_requirements": ["mas", "cert", "internal_security"],
                "remediation_timeline_hours": 72
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version="1.0.0",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        self.add_policy(mas_policy)
    
    def add_policy(self, policy: CompliancePolicy) -> bool:
        """Add a compliance policy to the engine"""
        try:
            policy_hash = self._calculate_policy_hash(policy)
            policy.policy_id = f"{policy.policy_id}-{policy_hash[:8]}"
            
            self.policies[policy.policy_id] = policy
            
            # Update jurisdiction index
            if policy.jurisdiction not in self.jurisdiction_policies:
                self.jurisdiction_policies[policy.jurisdiction] = []
            self.jurisdiction_policies[policy.jurisdiction].append(policy.policy_id)
            
            # Update framework index
            if policy.framework not in self.framework_policies:
                self.framework_policies[policy.framework] = []
            self.framework_policies[policy.framework].append(policy.policy_id)
            
            # Persist to storage
            self._save_policy(policy)
            
            self.logger.info(f"Added compliance policy: {policy.name} ({policy.policy_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add policy {policy.policy_id}: {str(e)}")
            return False
    
    def get_applicable_policies(self, 
                              jurisdiction: Optional[JurisdictionType] = None,
                              framework: Optional[ComplianceFramework] = None,
                              data_classification: Optional[DataClassification] = None) -> List[CompliancePolicy]:
        """Get applicable policies based on criteria"""
        applicable_policies = []
        
        for policy in self.policies.values():
            if jurisdiction and policy.jurisdiction != jurisdiction:
                continue
            if framework and policy.framework != framework:
                continue
            if data_classification and policy.data_classification != data_classification:
                continue
            
            # Check if policy is currently effective
            now = datetime.now(timezone.utc)
            if policy.effective_date > now:
                continue
            if policy.expiry_date and policy.expiry_date <= now:
                continue
            
            applicable_policies.append(policy)
        
        return applicable_policies
    
    def evaluate_compliance(self, 
                          event: Dict[str, Any],
                          jurisdiction: JurisdictionType,
                          data_classification: DataClassification) -> Dict[str, Any]:
        """Evaluate compliance for a specific event"""
        
        evaluation_id = str(uuid.uuid4())
        evaluation_start = time.time()
        
        applicable_policies = self.get_applicable_policies(
            jurisdiction=jurisdiction,
            data_classification=data_classification
        )
        
        violations = []
        warnings = []
        compliance_score = 100.0
        
        for policy in applicable_policies:
            policy_result = self._evaluate_policy(policy, event)
            
            if policy_result["violations"]:
                violations.extend(policy_result["violations"])
                compliance_score -= policy_result["penalty_score"]
            
            if policy_result["warnings"]:
                warnings.extend(policy_result["warnings"])
                compliance_score -= policy_result["warning_penalty"]
        
        evaluation_time = time.time() - evaluation_start
        
        result = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jurisdiction": jurisdiction.value,
            "data_classification": data_classification.value,
            "applicable_policies_count": len(applicable_policies),
            "compliance_score": max(0, compliance_score),
            "violations": violations,
            "warnings": warnings,
            "is_compliant": len(violations) == 0,
            "risk_level": self._calculate_risk_level(violations),
            "evaluation_time_seconds": evaluation_time,
            "recommendations": self._generate_recommendations(violations, warnings)
        }
        
        if violations:
            self._handle_violations(violations, evaluation_id)
        
        return result
    
    def _evaluate_policy(self, policy: CompliancePolicy, event: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific policy against an event"""
        violations = []
        warnings = []
        penalty_score = 0
        warning_penalty = 0
        
        # Data classification validation
        event_classification = event.get("data_classification", "internal")
        if DataClassification(event_classification).value != policy.data_classification.value:
            if self._is_higher_classification(event_classification, policy.data_classification.value):
                violations.append({
                    "policy_id": policy.policy_id,
                    "violation_type": "data_classification_mismatch",
                    "severity": "high",
                    "description": f"Event classified as {event_classification} but policy requires {policy.data_classification.value}",
                    "requirement": "Data classification compliance"
                })
                penalty_score += 20
        
        # Cross-border transfer validation
        if event.get("cross_border_transfer", False):
            source_country = event.get("source_country", "unknown")
            destination_country = event.get("destination_country", "unknown")
            
            for restriction in policy.cross_border_restrictions:
                if self._violates_transfer_restriction(restriction, source_country, destination_country):
                    violations.append({
                        "policy_id": policy.policy_id,
                        "violation_type": "cross_border_restriction",
                        "severity": "critical",
                        "description": f"Cross-border transfer violates restriction: {restriction}",
                        "requirement": "Cross-border data transfer compliance"
                    })
                    penalty_score += 50
        
        # Retention period validation
        if event.get("data_age_days", 0) > policy.retention_period_days:
            violations.append({
                "policy_id": policy.policy_id,
                "violation_type": "retention_period_exceeded",
                "severity": "medium",
                "description": f"Data age {event.get('data_age_days')} days exceeds policy limit {policy.retention_period_days} days",
                "requirement": "Data retention period compliance"
            })
            penalty_score += 15
        
        # Access control validation
        if event.get("access_type") == "administrative" and not event.get("multi_factor_auth", False):
            warnings.append({
                "policy_id": policy.policy_id,
                "warning_type": "access_control",
                "severity": "medium",
                "description": "Administrative access without multi-factor authentication",
                "recommendation": "Enable MFA for all administrative access"
            })
            warning_penalty += 5
        
        return {
            "violations": violations,
            "warnings": warnings,
            "penalty_score": penalty_score,
            "warning_penalty": warning_penalty
        }
    
    def _is_higher_classification(self, event_classification: str, policy_classification: str) -> bool:
        """Check if event classification is higher than policy classification"""
        classifications = ["public", "internal", "confidential", "restricted", "top_secret"]
        return classifications.index(event_classification) > classifications.index(policy_classification)
    
    def _violates_transfer_restriction(self, restriction: str, source: str, destination: str) -> bool:
        """Check if cross-border transfer violates a specific restriction"""
        # Simplified logic - in production, this would be more sophisticated
        if "non-adequate countries" in restriction.lower():
            non_adequate_countries = ["china", "russia", "north_korea", "iran"]
            return destination.lower() in non_adequate_countries
        return False
    
    def _calculate_risk_level(self, violations: List[Dict]) -> str:
        """Calculate overall risk level based on violations"""
        if not violations:
            return "low"
        
        critical_count = sum(1 for v in violations if v.get("severity") == "critical")
        high_count = sum(1 for v in violations if v.get("severity") == "high")
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif high_count > 0:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, violations: List[Dict], warnings: List[Dict]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        violation_types = set(v.get("violation_type") for v in violations)
        
        if "data_classification_mismatch" in violation_types:
            recommendations.append("Review and update data classification procedures")
        
        if "cross_border_restriction" in violation_types:
            recommendations.append("Implement data transfer approval workflows")
        
        if "retention_period_exceeded" in violation_types:
            recommendations.append("Establish automated data lifecycle management")
        
        if warnings:
            recommendations.append("Address security warnings to prevent future violations")
        
        return recommendations
    
    def _handle_violations(self, violations: List[Dict], evaluation_id: str):
        """Handle compliance violations"""
        for violation in violations:
            violation_id = f"VIOL-{evaluation_id[:8]}-{uuid.uuid4().hex[:8]}"
            self.active_violations.add(violation_id)
            
            self.logger.critical(f"Compliance violation detected: {violation_id}")
            self.logger.critical(f"Policy: {violation['policy_id']}")
            self.logger.critical(f"Type: {violation['violation_type']}")
            self.logger.critical(f"Severity: {violation['severity']}")
            self.logger.critical(f"Description: {violation['description']}")
    
    def _calculate_policy_hash(self, policy: CompliancePolicy) -> str:
        """Calculate hash for policy integrity verification"""
        policy_dict = asdict(policy)
        policy_dict.pop("created_at", None)
        policy_dict.pop("updated_at", None)
        policy_json = json.dumps(policy_dict, sort_keys=True)
        return hashlib.sha256(policy_json.encode()).hexdigest()
    
    def _save_policy(self, policy: CompliancePolicy):
        """Save policy to persistent storage"""
        policy_file = self.data_directory / f"{policy.policy_id}.json"
        policy_dict = asdict(policy)
        
        # Convert datetime objects to ISO format
        for key, value in policy_dict.items():
            if isinstance(value, datetime):
                policy_dict[key] = value.isoformat()
        
        with open(policy_file, 'w') as f:
            json.dump(policy_dict, f, indent=2, default=str)
    
    async def monitor_compliance(self):
        """Continuous compliance monitoring"""
        while True:
            try:
                # Check for policy updates
                await self._check_policy_updates()
                
                # Validate active violations
                await self._validate_active_violations()
                
                # Generate compliance metrics
                await self._generate_compliance_metrics()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_policy_updates(self):
        """Check for policy updates from regulatory sources"""
        # In production, this would check regulatory API endpoints
        pass
    
    async def _validate_active_violations(self):
        """Validate and clean up resolved violations"""
        # In production, this would check if violations have been resolved
        pass
    
    async def _generate_compliance_metrics(self):
        """Generate compliance metrics for monitoring"""
        metrics = {
            "total_policies": len(self.policies),
            "active_violations": len(self.active_violations),
            "jurisdiction_coverage": len(self.jurisdiction_policies),
            "framework_coverage": len(self.framework_policies)
        }
        
        self.logger.info(f"Compliance metrics: {metrics}")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_policies": len(self.policies),
            "active_violations": len(self.active_violations),
            "jurisdictions": list(j.value for j in self.jurisdiction_policies.keys()),
            "frameworks": list(f.value for f in self.framework_policies.keys()),
            "is_compliant": len(self.active_violations) == 0,
            "last_evaluation": datetime.now(timezone.utc).isoformat()
        }
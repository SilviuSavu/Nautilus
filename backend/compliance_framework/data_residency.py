"""
Data Residency and Classification System
=======================================

Advanced data governance system that enforces data residency requirements
and classification policies across multiple jurisdictions.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import uuid
import re
import hashlib


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class DataCategory(Enum):
    """Categories of data for compliance purposes"""
    PERSONAL_DATA = "personal_data"
    FINANCIAL_DATA = "financial_data"
    TRADING_DATA = "trading_data"
    SYSTEM_DATA = "system_data"
    CONFIGURATION_DATA = "configuration_data"
    SECURITY_DATA = "security_data"
    METADATA = "metadata"
    DERIVED_DATA = "derived_data"


class Jurisdiction(Enum):
    """Supported jurisdictions with data residency requirements"""
    EU = "eu"
    US = "us"
    UK = "uk"
    CANADA = "canada"
    SINGAPORE = "singapore"
    HONG_KONG = "hong_kong"
    JAPAN = "japan"
    AUSTRALIA = "australia"
    SWITZERLAND = "switzerland"
    GLOBAL = "global"


class DataLocation(Enum):
    """Physical data storage locations"""
    US_EAST = "us_east"
    US_WEST = "us_west"
    EU_WEST = "eu_west"
    EU_CENTRAL = "eu_central"
    UK_SOUTH = "uk_south"
    ASIA_PACIFIC = "asia_pacific"
    SINGAPORE = "singapore"
    AUSTRALIA = "australia"
    CANADA = "canada"
    MULTI_REGION = "multi_region"


class EncryptionLevel(Enum):
    """Data encryption requirements"""
    NONE = "none"
    STANDARD = "standard"
    HIGH = "high"
    QUANTUM_RESISTANT = "quantum_resistant"


class TransferMechanism(Enum):
    """Legal mechanisms for cross-border data transfers"""
    ADEQUACY_DECISION = "adequacy_decision"
    STANDARD_CONTRACTUAL_CLAUSES = "standard_contractual_clauses"
    BINDING_CORPORATE_RULES = "binding_corporate_rules"
    CERTIFICATION = "certification"
    CODES_OF_CONDUCT = "codes_of_conduct"
    DEROGATION = "derogation"
    NOT_REQUIRED = "not_required"


@dataclass
class DataRecord:
    """Individual data record with classification and residency information"""
    record_id: str
    data_category: DataCategory
    classification: DataClassification
    subject_jurisdiction: Jurisdiction
    data_location: DataLocation
    encryption_level: EncryptionLevel
    created_at: datetime
    updated_at: datetime
    retention_period_days: int
    deletion_date: Optional[datetime]
    cross_border_transfers: List[str]
    access_log: List[str]
    compliance_tags: Set[str]
    business_purpose: str
    legal_basis: str
    data_subject_id: Optional[str]
    pseudonymized: bool
    anonymized: bool
    backup_locations: List[DataLocation]
    metadata: Dict[str, Any]


@dataclass
class ResidencyRule:
    """Data residency rule definition"""
    rule_id: str
    name: str
    jurisdiction: Jurisdiction
    data_category: DataCategory
    classification: DataClassification
    required_location: DataLocation
    allowed_locations: List[DataLocation]
    prohibited_locations: List[DataLocation]
    encryption_required: EncryptionLevel
    cross_border_restrictions: Dict[str, Any]
    transfer_mechanisms: List[TransferMechanism]
    retention_requirements: Dict[str, Any]
    access_restrictions: Dict[str, Any]
    compliance_framework: str
    regulatory_body: str
    violation_penalty: str
    last_updated: datetime
    effective_date: datetime
    expiry_date: Optional[datetime]
    is_active: bool


@dataclass
class TransferApproval:
    """Cross-border data transfer approval record"""
    approval_id: str
    transfer_id: str
    source_jurisdiction: Jurisdiction
    destination_jurisdiction: Jurisdiction
    data_category: DataCategory
    classification: DataClassification
    transfer_mechanism: TransferMechanism
    legal_basis: str
    business_justification: str
    approved_by: str
    approval_date: datetime
    expiry_date: datetime
    conditions: List[str]
    safeguards: List[str]
    is_active: bool
    audit_trail: List[Dict[str, Any]]


class DataResidencyController:
    """
    Advanced data residency and classification controller that enforces
    jurisdiction-specific data governance requirements.
    """
    
    def __init__(self, data_directory: str = "/app/compliance/residency"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        self.data_records: Dict[str, DataRecord] = {}
        self.residency_rules: Dict[str, ResidencyRule] = {}
        self.transfer_approvals: Dict[str, TransferApproval] = {}
        
        # Indexes for fast lookup
        self.jurisdiction_index: Dict[Jurisdiction, Set[str]] = {}
        self.location_index: Dict[DataLocation, Set[str]] = {}
        self.classification_index: Dict[DataClassification, Set[str]] = {}
        
        self.logger = logging.getLogger("compliance.data_residency")
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_compliance())
        asyncio.create_task(self._cleanup_expired_records())
    
    def _initialize_default_rules(self):
        """Initialize default data residency rules for various jurisdictions"""
        
        # EU GDPR Personal Data Rule
        eu_personal_rule = ResidencyRule(
            rule_id="EU-GDPR-001",
            name="EU Personal Data Residency",
            jurisdiction=Jurisdiction.EU,
            data_category=DataCategory.PERSONAL_DATA,
            classification=DataClassification.CONFIDENTIAL,
            required_location=DataLocation.EU_WEST,
            allowed_locations=[DataLocation.EU_WEST, DataLocation.EU_CENTRAL],
            prohibited_locations=[DataLocation.US_EAST, DataLocation.US_WEST, DataLocation.ASIA_PACIFIC],
            encryption_required=EncryptionLevel.HIGH,
            cross_border_restrictions={
                "adequacy_countries": ["canada", "uk", "switzerland"],
                "requires_safeguards": ["us", "singapore", "australia"],
                "prohibited_countries": ["china", "russia"]
            },
            transfer_mechanisms=[
                TransferMechanism.ADEQUACY_DECISION,
                TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES,
                TransferMechanism.BINDING_CORPORATE_RULES
            ],
            retention_requirements={
                "maximum_period_days": 2555,  # 7 years
                "deletion_required": True,
                "anonymization_allowed": True
            },
            access_restrictions={
                "require_mfa": True,
                "audit_all_access": True,
                "role_based_access": True
            },
            compliance_framework="GDPR",
            regulatory_body="European Data Protection Board",
            violation_penalty="Up to 4% of annual turnover or €20M",
            last_updated=datetime.now(timezone.utc),
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            expiry_date=None,
            is_active=True
        )
        self.add_residency_rule(eu_personal_rule)
        
        # US Financial Data Rule (FINRA/SEC)
        us_financial_rule = ResidencyRule(
            rule_id="US-FINRA-001",
            name="US Financial Data Residency",
            jurisdiction=Jurisdiction.US,
            data_category=DataCategory.FINANCIAL_DATA,
            classification=DataClassification.RESTRICTED,
            required_location=DataLocation.US_EAST,
            allowed_locations=[DataLocation.US_EAST, DataLocation.US_WEST],
            prohibited_locations=[DataLocation.EU_WEST, DataLocation.ASIA_PACIFIC],
            encryption_required=EncryptionLevel.HIGH,
            cross_border_restrictions={
                "requires_regulatory_approval": True,
                "prohibited_jurisdictions": ["china", "iran", "north_korea", "russia"],
                "backup_restrictions": "us_only"
            },
            transfer_mechanisms=[TransferMechanism.DEROGATION],
            retention_requirements={
                "minimum_period_days": 2190,  # 6 years (FINRA requirement)
                "deletion_prohibited": True,
                "archival_required": True
            },
            access_restrictions={
                "us_persons_only": True,
                "security_clearance": "confidential",
                "audit_trail_required": True
            },
            compliance_framework="FINRA Rule 4511",
            regulatory_body="FINRA/SEC",
            violation_penalty="Civil and criminal penalties",
            last_updated=datetime.now(timezone.utc),
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            expiry_date=None,
            is_active=True
        )
        self.add_residency_rule(us_financial_rule)
        
        # Singapore Banking Data Rule
        sg_banking_rule = ResidencyRule(
            rule_id="SG-MAS-001",
            name="Singapore Banking Data Residency",
            jurisdiction=Jurisdiction.SINGAPORE,
            data_category=DataCategory.FINANCIAL_DATA,
            classification=DataClassification.CONFIDENTIAL,
            required_location=DataLocation.SINGAPORE,
            allowed_locations=[DataLocation.SINGAPORE, DataLocation.ASIA_PACIFIC],
            prohibited_locations=[],
            encryption_required=EncryptionLevel.HIGH,
            cross_border_restrictions={
                "requires_mas_approval": True,
                "acceptable_jurisdictions": ["eu", "us", "uk", "australia"],
                "risk_assessment_required": True
            },
            transfer_mechanisms=[
                TransferMechanism.ADEQUACY_DECISION,
                TransferMechanism.CERTIFICATION
            ],
            retention_requirements={
                "minimum_period_days": 2190,  # 6 years
                "mas_approval_for_deletion": True
            },
            access_restrictions={
                "singapore_oversight": True,
                "local_management": True
            },
            compliance_framework="MAS Technology Risk Management",
            regulatory_body="Monetary Authority of Singapore",
            violation_penalty="Up to S$1M fine",
            last_updated=datetime.now(timezone.utc),
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            expiry_date=None,
            is_active=True
        )
        self.add_residency_rule(sg_banking_rule)
        
        # UK Data Protection Rule
        uk_data_rule = ResidencyRule(
            rule_id="UK-DPA-001",
            name="UK Data Protection Act Compliance",
            jurisdiction=Jurisdiction.UK,
            data_category=DataCategory.PERSONAL_DATA,
            classification=DataClassification.CONFIDENTIAL,
            required_location=DataLocation.UK_SOUTH,
            allowed_locations=[DataLocation.UK_SOUTH, DataLocation.EU_WEST],
            prohibited_locations=[],
            encryption_required=EncryptionLevel.STANDARD,
            cross_border_restrictions={
                "adequacy_countries": ["eu", "canada", "switzerland"],
                "brexit_considerations": True
            },
            transfer_mechanisms=[
                TransferMechanism.ADEQUACY_DECISION,
                TransferMechanism.STANDARD_CONTRACTUAL_CLAUSES
            ],
            retention_requirements={
                "maximum_period_days": 2555,  # 7 years
                "subject_rights_respected": True
            },
            access_restrictions={
                "uk_oversight": True,
                "ico_compliance": True
            },
            compliance_framework="UK DPA 2018",
            regulatory_body="Information Commissioner's Office",
            violation_penalty="Up to £17.5M or 4% of turnover",
            last_updated=datetime.now(timezone.utc),
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            expiry_date=None,
            is_active=True
        )
        self.add_residency_rule(uk_data_rule)
    
    def add_residency_rule(self, rule: ResidencyRule) -> bool:
        """Add a data residency rule"""
        try:
            self.residency_rules[rule.rule_id] = rule
            self.logger.info(f"Added residency rule: {rule.name} ({rule.rule_id})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add rule {rule.rule_id}: {str(e)}")
            return False
    
    def classify_data(self, 
                     data_content: str,
                     data_category: DataCategory,
                     subject_jurisdiction: Jurisdiction,
                     business_purpose: str,
                     legal_basis: str,
                     data_subject_id: Optional[str] = None) -> DataRecord:
        """Classify data and determine appropriate residency requirements"""
        
        record_id = str(uuid.uuid4())
        
        # Automatically determine classification based on content analysis
        classification = self._determine_classification(data_content, data_category)
        
        # Determine required location based on jurisdiction and classification
        required_location = self._determine_location(subject_jurisdiction, data_category, classification)
        
        # Determine encryption requirements
        encryption_level = self._determine_encryption(classification, data_category)
        
        # Calculate retention period
        retention_days = self._calculate_retention(data_category, subject_jurisdiction)
        
        # Create data record
        record = DataRecord(
            record_id=record_id,
            data_category=data_category,
            classification=classification,
            subject_jurisdiction=subject_jurisdiction,
            data_location=required_location,
            encryption_level=encryption_level,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            retention_period_days=retention_days,
            deletion_date=datetime.now(timezone.utc) + timedelta(days=retention_days),
            cross_border_transfers=[],
            access_log=[],
            compliance_tags=set(),
            business_purpose=business_purpose,
            legal_basis=legal_basis,
            data_subject_id=data_subject_id,
            pseudonymized=self._is_pseudonymized(data_content),
            anonymized=self._is_anonymized(data_content),
            backup_locations=self._determine_backup_locations(required_location),
            metadata={}
        )
        
        # Store record
        self.data_records[record_id] = record
        
        # Update indexes
        self._update_indexes(record)
        
        self.logger.info(f"Classified data record: {record_id} ({classification.value})")
        return record
    
    def _determine_classification(self, content: str, category: DataCategory) -> DataClassification:
        """Determine data classification based on content analysis"""
        
        # High-risk patterns
        high_risk_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card numbers
            r'password|secret|private_key',
            r'confidential|restricted|classified'
        ]
        
        # Medium-risk patterns
        medium_risk_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone numbers
            r'internal|proprietary'
        ]
        
        content_lower = content.lower()
        
        # Check for high-risk content
        for pattern in high_risk_patterns:
            if re.search(pattern, content_lower):
                if category in [DataCategory.PERSONAL_DATA, DataCategory.FINANCIAL_DATA]:
                    return DataClassification.RESTRICTED
                else:
                    return DataClassification.CONFIDENTIAL
        
        # Check for medium-risk content
        for pattern in medium_risk_patterns:
            if re.search(pattern, content_lower):
                return DataClassification.CONFIDENTIAL
        
        # Default classification based on category
        if category == DataCategory.PERSONAL_DATA:
            return DataClassification.CONFIDENTIAL
        elif category == DataCategory.FINANCIAL_DATA:
            return DataClassification.RESTRICTED
        elif category == DataCategory.SECURITY_DATA:
            return DataClassification.CONFIDENTIAL
        else:
            return DataClassification.INTERNAL
    
    def _determine_location(self, jurisdiction: Jurisdiction, 
                          category: DataCategory, 
                          classification: DataClassification) -> DataLocation:
        """Determine required storage location based on jurisdiction and data type"""
        
        # Find applicable rules
        applicable_rules = []
        for rule in self.residency_rules.values():
            if (rule.jurisdiction == jurisdiction and 
                rule.data_category == category and
                rule.classification == classification and
                rule.is_active):
                applicable_rules.append(rule)
        
        if applicable_rules:
            # Use the most restrictive rule
            most_restrictive = min(applicable_rules, 
                                 key=lambda r: len(r.allowed_locations))
            return most_restrictive.required_location
        
        # Default location mapping
        location_mapping = {
            Jurisdiction.EU: DataLocation.EU_WEST,
            Jurisdiction.US: DataLocation.US_EAST,
            Jurisdiction.UK: DataLocation.UK_SOUTH,
            Jurisdiction.SINGAPORE: DataLocation.SINGAPORE,
            Jurisdiction.CANADA: DataLocation.CANADA,
            Jurisdiction.AUSTRALIA: DataLocation.AUSTRALIA,
            Jurisdiction.HONG_KONG: DataLocation.ASIA_PACIFIC,
            Jurisdiction.JAPAN: DataLocation.ASIA_PACIFIC
        }
        
        return location_mapping.get(jurisdiction, DataLocation.US_EAST)
    
    def _determine_encryption(self, classification: DataClassification, 
                            category: DataCategory) -> EncryptionLevel:
        """Determine required encryption level"""
        
        if classification == DataClassification.RESTRICTED:
            return EncryptionLevel.QUANTUM_RESISTANT
        elif classification == DataClassification.CONFIDENTIAL:
            return EncryptionLevel.HIGH
        elif category in [DataCategory.PERSONAL_DATA, DataCategory.FINANCIAL_DATA]:
            return EncryptionLevel.HIGH
        else:
            return EncryptionLevel.STANDARD
    
    def _calculate_retention(self, category: DataCategory, jurisdiction: Jurisdiction) -> int:
        """Calculate retention period in days"""
        
        # Jurisdiction-specific retention requirements
        if jurisdiction == Jurisdiction.EU:
            if category == DataCategory.PERSONAL_DATA:
                return 2555  # 7 years (common business practice)
            elif category == DataCategory.FINANCIAL_DATA:
                return 2555  # 7 years
        
        elif jurisdiction == Jurisdiction.US:
            if category == DataCategory.FINANCIAL_DATA:
                return 2190  # 6 years (FINRA requirement)
            elif category == DataCategory.TRADING_DATA:
                return 2190  # 6 years
        
        elif jurisdiction == Jurisdiction.SINGAPORE:
            if category == DataCategory.FINANCIAL_DATA:
                return 2190  # 6 years (MAS requirement)
        
        # Default retention periods
        category_defaults = {
            DataCategory.PERSONAL_DATA: 2555,  # 7 years
            DataCategory.FINANCIAL_DATA: 2190,  # 6 years
            DataCategory.TRADING_DATA: 2190,   # 6 years
            DataCategory.SYSTEM_DATA: 1095,    # 3 years
            DataCategory.SECURITY_DATA: 1095,  # 3 years
            DataCategory.METADATA: 365,        # 1 year
            DataCategory.CONFIGURATION_DATA: 1095,  # 3 years
            DataCategory.DERIVED_DATA: 1095    # 3 years
        }
        
        return category_defaults.get(category, 1095)
    
    def _is_pseudonymized(self, content: str) -> bool:
        """Check if data appears to be pseudonymized"""
        # Look for patterns that suggest pseudonymization
        pseudonym_patterns = [
            r'\b[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}\b',  # UUID
            r'\b[a-f0-9]{32}\b',  # MD5 hash
            r'\b[a-f0-9]{64}\b',  # SHA256 hash
            r'USER_[0-9]+',       # User ID pattern
        ]
        
        for pattern in pseudonym_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _is_anonymized(self, content: str) -> bool:
        """Check if data appears to be anonymized"""
        # Look for patterns that suggest anonymization
        anon_patterns = [
            r'\*{3,}',           # Masked data
            r'[X]{3,}',          # X'ed out data
            r'REDACTED',         # Explicit redaction
            r'ANONYMIZED',       # Explicit anonymization
        ]
        
        for pattern in anon_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _determine_backup_locations(self, primary_location: DataLocation) -> List[DataLocation]:
        """Determine appropriate backup locations"""
        
        # Backup location rules to ensure compliance
        backup_rules = {
            DataLocation.US_EAST: [DataLocation.US_WEST],
            DataLocation.US_WEST: [DataLocation.US_EAST],
            DataLocation.EU_WEST: [DataLocation.EU_CENTRAL],
            DataLocation.EU_CENTRAL: [DataLocation.EU_WEST],
            DataLocation.UK_SOUTH: [DataLocation.EU_WEST],
            DataLocation.SINGAPORE: [DataLocation.ASIA_PACIFIC],
            DataLocation.ASIA_PACIFIC: [DataLocation.SINGAPORE],
            DataLocation.AUSTRALIA: [DataLocation.ASIA_PACIFIC],
            DataLocation.CANADA: [DataLocation.US_EAST]
        }
        
        return backup_rules.get(primary_location, [])
    
    def _update_indexes(self, record: DataRecord):
        """Update lookup indexes"""
        
        # Jurisdiction index
        if record.subject_jurisdiction not in self.jurisdiction_index:
            self.jurisdiction_index[record.subject_jurisdiction] = set()
        self.jurisdiction_index[record.subject_jurisdiction].add(record.record_id)
        
        # Location index
        if record.data_location not in self.location_index:
            self.location_index[record.data_location] = set()
        self.location_index[record.data_location].add(record.record_id)
        
        # Classification index
        if record.classification not in self.classification_index:
            self.classification_index[record.classification] = set()
        self.classification_index[record.classification].add(record.record_id)
    
    def validate_transfer(self, 
                         record_id: str,
                         destination_jurisdiction: Jurisdiction,
                         transfer_purpose: str) -> Dict[str, Any]:
        """Validate if a cross-border data transfer is compliant"""
        
        if record_id not in self.data_records:
            return {"valid": False, "error": "Record not found"}
        
        record = self.data_records[record_id]
        
        # Find applicable residency rules
        applicable_rules = [
            rule for rule in self.residency_rules.values()
            if (rule.jurisdiction == record.subject_jurisdiction and
                rule.data_category == record.data_category and
                rule.classification == record.classification and
                rule.is_active)
        ]
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "requirements": [],
            "prohibited": False,
            "transfer_mechanisms": [],
            "safeguards_required": []
        }
        
        for rule in applicable_rules:
            # Check if destination is prohibited
            restrictions = rule.cross_border_restrictions
            prohibited_countries = restrictions.get("prohibited_countries", [])
            
            if destination_jurisdiction.value in prohibited_countries:
                validation_result["valid"] = False
                validation_result["prohibited"] = True
                validation_result["warnings"].append(
                    f"Transfer to {destination_jurisdiction.value} is prohibited by {rule.name}"
                )
                continue
            
            # Check if safeguards are required
            requires_safeguards = restrictions.get("requires_safeguards", [])
            if destination_jurisdiction.value in requires_safeguards:
                validation_result["safeguards_required"].extend([
                    "Standard Contractual Clauses",
                    "Adequacy Assessment",
                    "Data Subject Consent"
                ])
            
            # Add available transfer mechanisms
            validation_result["transfer_mechanisms"].extend(
                [tm.value for tm in rule.transfer_mechanisms]
            )
            
            # Add requirements
            if restrictions.get("requires_regulatory_approval", False):
                validation_result["requirements"].append("Regulatory approval required")
            
            if restrictions.get("risk_assessment_required", False):
                validation_result["requirements"].append("Risk assessment required")
        
        return validation_result
    
    def request_transfer_approval(self,
                                 record_id: str,
                                 destination_jurisdiction: Jurisdiction,
                                 transfer_mechanism: TransferMechanism,
                                 business_justification: str,
                                 requested_by: str) -> Optional[str]:
        """Request approval for cross-border data transfer"""
        
        if record_id not in self.data_records:
            return None
        
        record = self.data_records[record_id]
        
        # Validate transfer first
        validation = self.validate_transfer(record_id, destination_jurisdiction, business_justification)
        if not validation["valid"]:
            self.logger.error(f"Transfer validation failed for {record_id}")
            return None
        
        # Create approval request
        approval_id = str(uuid.uuid4())
        
        approval = TransferApproval(
            approval_id=approval_id,
            transfer_id=str(uuid.uuid4()),
            source_jurisdiction=record.subject_jurisdiction,
            destination_jurisdiction=destination_jurisdiction,
            data_category=record.data_category,
            classification=record.classification,
            transfer_mechanism=transfer_mechanism,
            legal_basis=record.legal_basis,
            business_justification=business_justification,
            approved_by=requested_by,
            approval_date=datetime.now(timezone.utc),
            expiry_date=datetime.now(timezone.utc) + timedelta(days=365),  # 1 year validity
            conditions=validation["requirements"],
            safeguards=validation["safeguards_required"],
            is_active=True,
            audit_trail=[{
                "action": "approval_created",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": requested_by
            }]
        )
        
        self.transfer_approvals[approval_id] = approval
        
        # Update record
        record.cross_border_transfers.append(approval_id)
        record.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Transfer approval created: {approval_id}")
        return approval_id
    
    def get_compliance_status(self, 
                             jurisdiction: Optional[Jurisdiction] = None,
                             data_category: Optional[DataCategory] = None) -> Dict[str, Any]:
        """Get compliance status overview"""
        
        total_records = len(self.data_records)
        
        # Filter records if criteria provided
        filtered_records = self.data_records.values()
        if jurisdiction:
            record_ids = self.jurisdiction_index.get(jurisdiction, set())
            filtered_records = [r for r in filtered_records if r.record_id in record_ids]
        if data_category:
            filtered_records = [r for r in filtered_records if r.data_category == data_category]
        
        # Calculate statistics
        location_distribution = {}
        classification_distribution = {}
        encryption_distribution = {}
        
        for record in filtered_records:
            # Location distribution
            loc = record.data_location.value
            location_distribution[loc] = location_distribution.get(loc, 0) + 1
            
            # Classification distribution
            cls = record.classification.value
            classification_distribution[cls] = classification_distribution.get(cls, 0) + 1
            
            # Encryption distribution
            enc = record.encryption_level.value
            encryption_distribution[enc] = encryption_distribution.get(enc, 0) + 1
        
        # Check for violations
        violations = self._check_compliance_violations(list(filtered_records))
        
        return {
            "total_records": len(filtered_records),
            "jurisdiction": jurisdiction.value if jurisdiction else "all",
            "data_category": data_category.value if data_category else "all",
            "location_distribution": location_distribution,
            "classification_distribution": classification_distribution,
            "encryption_distribution": encryption_distribution,
            "compliance_violations": len(violations),
            "violation_details": violations,
            "active_transfer_approvals": len([a for a in self.transfer_approvals.values() if a.is_active]),
            "last_assessment": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_compliance_violations(self, records: List[DataRecord]) -> List[Dict[str, Any]]:
        """Check for compliance violations in data records"""
        
        violations = []
        
        for record in records:
            # Check location compliance
            applicable_rules = [
                rule for rule in self.residency_rules.values()
                if (rule.jurisdiction == record.subject_jurisdiction and
                    rule.data_category == record.data_category and
                    rule.classification == record.classification and
                    rule.is_active)
            ]
            
            for rule in applicable_rules:
                # Check location violation
                if record.data_location not in rule.allowed_locations:
                    violations.append({
                        "record_id": record.record_id,
                        "violation_type": "location_violation",
                        "rule_id": rule.rule_id,
                        "description": f"Data located in {record.data_location.value} but rule requires {rule.allowed_locations}",
                        "severity": "high"
                    })
                
                # Check encryption requirement
                required_encryption = rule.encryption_required
                if self._encryption_level_priority(record.encryption_level) < self._encryption_level_priority(required_encryption):
                    violations.append({
                        "record_id": record.record_id,
                        "violation_type": "encryption_violation",
                        "rule_id": rule.rule_id,
                        "description": f"Encryption level {record.encryption_level.value} insufficient, requires {required_encryption.value}",
                        "severity": "medium"
                    })
                
                # Check retention violation
                if record.deletion_date and record.deletion_date < datetime.now(timezone.utc):
                    if rule.retention_requirements.get("deletion_prohibited", False):
                        violations.append({
                            "record_id": record.record_id,
                            "violation_type": "retention_violation",
                            "rule_id": rule.rule_id,
                            "description": "Data scheduled for deletion but deletion is prohibited",
                            "severity": "critical"
                        })
        
        return violations
    
    def _encryption_level_priority(self, level: EncryptionLevel) -> int:
        """Get numeric priority for encryption level comparison"""
        priorities = {
            EncryptionLevel.NONE: 0,
            EncryptionLevel.STANDARD: 1,
            EncryptionLevel.HIGH: 2,
            EncryptionLevel.QUANTUM_RESISTANT: 3
        }
        return priorities.get(level, 0)
    
    async def _monitor_compliance(self):
        """Continuous compliance monitoring task"""
        while True:
            try:
                # Check for violations
                all_records = list(self.data_records.values())
                violations = self._check_compliance_violations(all_records)
                
                if violations:
                    self.logger.warning(f"Found {len(violations)} compliance violations")
                    for violation in violations:
                        self.logger.warning(f"Violation: {violation}")
                
                # Check expired transfer approvals
                expired_approvals = [
                    approval for approval in self.transfer_approvals.values()
                    if approval.is_active and approval.expiry_date < datetime.now(timezone.utc)
                ]
                
                for approval in expired_approvals:
                    approval.is_active = False
                    self.logger.info(f"Transfer approval expired: {approval.approval_id}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring: {str(e)}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _cleanup_expired_records(self):
        """Cleanup task for expired data records"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                expired_records = []
                
                for record in self.data_records.values():
                    if record.deletion_date and record.deletion_date <= now:
                        expired_records.append(record.record_id)
                
                for record_id in expired_records:
                    # Check if deletion is allowed by jurisdiction rules
                    record = self.data_records[record_id]
                    can_delete = True
                    
                    applicable_rules = [
                        rule for rule in self.residency_rules.values()
                        if (rule.jurisdiction == record.subject_jurisdiction and
                            rule.data_category == record.data_category and
                            rule.is_active)
                    ]
                    
                    for rule in applicable_rules:
                        if rule.retention_requirements.get("deletion_prohibited", False):
                            can_delete = False
                            break
                    
                    if can_delete:
                        # Remove from indexes
                        self.jurisdiction_index[record.subject_jurisdiction].discard(record_id)
                        self.location_index[record.data_location].discard(record_id)
                        self.classification_index[record.classification].discard(record_id)
                        
                        # Remove record
                        del self.data_records[record_id]
                        
                        self.logger.info(f"Deleted expired record: {record_id}")
                    else:
                        self.logger.warning(f"Record {record_id} expired but deletion prohibited")
                
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def export_compliance_report(self, 
                                jurisdiction: Jurisdiction,
                                start_date: datetime,
                                end_date: datetime) -> Dict[str, Any]:
        """Export comprehensive compliance report for jurisdiction"""
        
        # Filter records by jurisdiction and date range
        jurisdiction_records = [
            record for record in self.data_records.values()
            if (record.subject_jurisdiction == jurisdiction and
                start_date <= record.created_at <= end_date)
        ]
        
        # Analyze compliance status
        violations = self._check_compliance_violations(jurisdiction_records)
        
        # Transfer analysis
        jurisdiction_approvals = [
            approval for approval in self.transfer_approvals.values()
            if approval.source_jurisdiction == jurisdiction
        ]
        
        return {
            "jurisdiction": jurisdiction.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_records": len(jurisdiction_records),
            "records_by_category": {
                category.value: len([r for r in jurisdiction_records if r.data_category == category])
                for category in DataCategory
            },
            "records_by_classification": {
                cls.value: len([r for r in jurisdiction_records if r.classification == cls])
                for cls in DataClassification
            },
            "compliance_violations": {
                "total": len(violations),
                "by_type": {
                    violation_type: len([v for v in violations if v["violation_type"] == violation_type])
                    for violation_type in set(v["violation_type"] for v in violations)
                }
            },
            "cross_border_transfers": {
                "total_approvals": len(jurisdiction_approvals),
                "active_approvals": len([a for a in jurisdiction_approvals if a.is_active]),
                "expired_approvals": len([a for a in jurisdiction_approvals if not a.is_active])
            },
            "data_locations": {
                location.value: len([r for r in jurisdiction_records if r.data_location == location])
                for location in DataLocation
            },
            "encryption_status": {
                level.value: len([r for r in jurisdiction_records if r.encryption_level == level])
                for level in EncryptionLevel
            },
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
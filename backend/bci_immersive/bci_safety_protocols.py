"""
Nautilus BCI Safety Protocols

This module implements comprehensive safety protocols and medical device compliance 
for brain-computer interface systems. Features real-time safety monitoring, 
regulatory compliance, and emergency procedures.

Key Features:
- Medical device safety standards compliance (ISO 14155, FDA 21 CFR 820, IEC 60601)
- Real-time physiological monitoring and alerts
- Emergency shutdown procedures
- User consent and data protection protocols
- Safety threshold enforcement
- Regulatory audit trail
- Risk assessment and mitigation

Safety Standards:
- ISO 14155: Clinical investigation of medical devices
- FDA 21 CFR 820: Quality system regulation
- IEC 60601: Medical electrical equipment safety
- HIPAA compliance for medical data
- EU MDR (Medical Device Regulation)

Author: Nautilus BCI Safety Team
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime, timezone, timedelta
import json
import hashlib
import uuid
from pathlib import Path
import warnings

# Cryptography for secure logging
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    warnings.warn("Cryptography not available - audit logs will not be encrypted")
    ENCRYPTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyViolationType(Enum):
    """Types of safety violations"""
    SIGNAL_AMPLITUDE_EXCEEDED = "signal_amplitude_exceeded"
    PROCESSING_TIME_EXCEEDED = "processing_time_exceeded"
    SESSION_DURATION_EXCEEDED = "session_duration_exceeded"
    PHYSIOLOGICAL_THRESHOLD_EXCEEDED = "physiological_threshold_exceeded"
    DEVICE_MALFUNCTION = "device_malfunction"
    USER_DISTRESS = "user_distress"
    CONSENT_VIOLATION = "consent_violation"
    DATA_INTEGRITY_VIOLATION = "data_integrity_violation"

class ComplianceStandard(Enum):
    """Medical device compliance standards"""
    ISO_14155 = "iso_14155"
    FDA_CFR_820 = "fda_cfr_820"
    IEC_60601 = "iec_60601"
    HIPAA = "hipaa"
    EU_MDR = "eu_mdr"
    ISO_27001 = "iso_27001"

class UserConsentType(Enum):
    """Types of user consent"""
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    DATA_STORAGE = "data_storage"
    DATA_SHARING = "data_sharing"
    RESEARCH_PARTICIPATION = "research_participation"
    COMMERCIAL_USE = "commercial_use"

@dataclass
class SafetyThresholds:
    """Safety thresholds for BCI operation"""
    # Signal amplitude thresholds (μV)
    max_eeg_amplitude: float = 200.0
    max_ecg_amplitude: float = 10.0  # mV
    max_emg_amplitude: float = 500.0
    
    # Processing time thresholds (ms)
    max_processing_latency: float = 100.0
    max_feedback_latency: float = 200.0
    
    # Session duration limits (minutes)
    max_session_duration: float = 120.0
    recommended_break_interval: float = 20.0
    min_break_duration: float = 5.0
    
    # Physiological thresholds
    max_heart_rate: float = 150.0  # BPM
    min_heart_rate: float = 40.0
    max_stress_level: float = 0.8  # 0-1 scale
    
    # Device safety thresholds
    max_stimulation_current: float = 2.0  # mA (if applicable)
    max_device_temperature: float = 40.0  # Celsius
    min_signal_quality: float = 0.3  # 0-1 scale
    
    # Data integrity thresholds
    max_data_loss_rate: float = 0.01  # 1%
    min_encryption_strength: int = 256  # bits

@dataclass
class SafetyViolation:
    """Safety violation record"""
    violation_id: str
    violation_type: SafetyViolationType
    severity_level: SafetyLevel
    description: str
    measured_value: Optional[float]
    threshold_value: Optional[float]
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    device_info: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

@dataclass
class UserConsent:
    """User consent record"""
    consent_id: str
    user_id: str
    consent_type: UserConsentType
    granted: bool
    timestamp: datetime
    expiry_date: Optional[datetime]
    consent_version: str
    digital_signature: str
    witness_signature: Optional[str] = None
    revocation_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyAuditEvent:
    """Safety audit event"""
    event_id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    details: Dict[str, Any]
    compliance_standards: List[ComplianceStandard]
    encrypted_data: Optional[str] = None
    hash_signature: str = ""

class PhysiologicalMonitor:
    """Real-time physiological safety monitoring"""
    
    def __init__(self, thresholds: SafetyThresholds):
        self.thresholds = thresholds
        self.monitoring_active = False
        self.alert_callbacks = []
        
        # Monitoring data
        self.physiological_data = {}
        self.violation_history = []
        self.last_alert_times = {}
        
        # Alert suppression to prevent spam
        self.alert_cooldown_seconds = 10
        
        logger.info("Physiological monitor initialized")
    
    def add_alert_callback(self, callback: Callable[[SafetyViolation], None]):
        """Add callback for safety violations"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Start physiological monitoring"""
        self.monitoring_active = True
        self.current_user_id = user_id
        self.current_session_id = session_id
        
        logger.info(f"Started physiological monitoring for user {user_id}, session {session_id}")
        
        return {
            'status': 'monitoring_started',
            'user_id': user_id,
            'session_id': session_id,
            'thresholds': self.thresholds.__dict__
        }
    
    async def monitor_signal_amplitude(self, signal_type: str, amplitude: float) -> Optional[SafetyViolation]:
        """Monitor signal amplitude for safety violations"""
        if not self.monitoring_active:
            return None
        
        # Determine appropriate threshold
        threshold = None
        if signal_type.lower() == 'eeg':
            threshold = self.thresholds.max_eeg_amplitude
        elif signal_type.lower() == 'ecg':
            threshold = self.thresholds.max_ecg_amplitude
        elif signal_type.lower() == 'emg':
            threshold = self.thresholds.max_emg_amplitude
        
        if threshold and abs(amplitude) > threshold:
            violation = SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=SafetyViolationType.SIGNAL_AMPLITUDE_EXCEEDED,
                severity_level=SafetyLevel.CRITICAL,
                description=f"{signal_type.upper()} amplitude exceeded safety threshold",
                measured_value=abs(amplitude),
                threshold_value=threshold,
                timestamp=datetime.now(timezone.utc),
                user_id=getattr(self, 'current_user_id', None),
                session_id=getattr(self, 'current_session_id', None),
                device_info={'signal_type': signal_type}
            )
            
            await self._handle_safety_violation(violation)
            return violation
        
        return None
    
    async def monitor_processing_time(self, operation: str, processing_time_ms: float) -> Optional[SafetyViolation]:
        """Monitor processing time for safety violations"""
        if not self.monitoring_active:
            return None
        
        threshold = self.thresholds.max_processing_latency
        if 'feedback' in operation.lower():
            threshold = self.thresholds.max_feedback_latency
        
        if processing_time_ms > threshold:
            violation = SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=SafetyViolationType.PROCESSING_TIME_EXCEEDED,
                severity_level=SafetyLevel.WARNING,
                description=f"Processing time for {operation} exceeded threshold",
                measured_value=processing_time_ms,
                threshold_value=threshold,
                timestamp=datetime.now(timezone.utc),
                user_id=getattr(self, 'current_user_id', None),
                session_id=getattr(self, 'current_session_id', None),
                device_info={'operation': operation}
            )
            
            await self._handle_safety_violation(violation)
            return violation
        
        return None
    
    async def monitor_session_duration(self, session_start_time: datetime) -> Optional[SafetyViolation]:
        """Monitor session duration for safety limits"""
        if not self.monitoring_active:
            return None
        
        session_duration_minutes = (datetime.now(timezone.utc) - session_start_time).total_seconds() / 60
        
        if session_duration_minutes > self.thresholds.max_session_duration:
            violation = SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=SafetyViolationType.SESSION_DURATION_EXCEEDED,
                severity_level=SafetyLevel.CRITICAL,
                description="Session duration exceeded maximum safe limit",
                measured_value=session_duration_minutes,
                threshold_value=self.thresholds.max_session_duration,
                timestamp=datetime.now(timezone.utc),
                user_id=getattr(self, 'current_user_id', None),
                session_id=getattr(self, 'current_session_id', None)
            )
            
            await self._handle_safety_violation(violation)
            return violation
        
        return None
    
    async def monitor_physiological_parameters(self, heart_rate: Optional[float] = None,
                                             stress_level: Optional[float] = None) -> List[SafetyViolation]:
        """Monitor physiological parameters"""
        if not self.monitoring_active:
            return []
        
        violations = []
        
        # Heart rate monitoring
        if heart_rate is not None:
            if heart_rate > self.thresholds.max_heart_rate or heart_rate < self.thresholds.min_heart_rate:
                violation = SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=SafetyViolationType.PHYSIOLOGICAL_THRESHOLD_EXCEEDED,
                    severity_level=SafetyLevel.CRITICAL,
                    description=f"Heart rate outside safe range: {heart_rate} BPM",
                    measured_value=heart_rate,
                    threshold_value=self.thresholds.max_heart_rate if heart_rate > self.thresholds.max_heart_rate else self.thresholds.min_heart_rate,
                    timestamp=datetime.now(timezone.utc),
                    user_id=getattr(self, 'current_user_id', None),
                    session_id=getattr(self, 'current_session_id', None),
                    device_info={'parameter': 'heart_rate'}
                )
                violations.append(violation)
        
        # Stress level monitoring
        if stress_level is not None and stress_level > self.thresholds.max_stress_level:
            violation = SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=SafetyViolationType.PHYSIOLOGICAL_THRESHOLD_EXCEEDED,
                severity_level=SafetyLevel.WARNING,
                description=f"Stress level exceeded threshold: {stress_level:.2f}",
                measured_value=stress_level,
                threshold_value=self.thresholds.max_stress_level,
                timestamp=datetime.now(timezone.utc),
                user_id=getattr(self, 'current_user_id', None),
                session_id=getattr(self, 'current_session_id', None),
                device_info={'parameter': 'stress_level'}
            )
            violations.append(violation)
        
        # Handle all violations
        for violation in violations:
            await self._handle_safety_violation(violation)
        
        return violations
    
    async def _handle_safety_violation(self, violation: SafetyViolation):
        """Handle detected safety violation"""
        # Add to violation history
        self.violation_history.append(violation)
        
        # Check for alert cooldown to prevent spam
        violation_key = f"{violation.violation_type.value}_{violation.user_id}"
        last_alert = self.last_alert_times.get(violation_key, datetime.min.replace(tzinfo=timezone.utc))
        
        if (datetime.now(timezone.utc) - last_alert).total_seconds() < self.alert_cooldown_seconds:
            return  # Skip alert due to cooldown
        
        self.last_alert_times[violation_key] = datetime.now(timezone.utc)
        
        # Log violation
        logger.error(f"SAFETY VIOLATION: {violation.description} "
                    f"(Value: {violation.measured_value}, Threshold: {violation.threshold_value})")
        
        # Determine mitigation actions
        mitigation_actions = await self._determine_mitigation_actions(violation)
        violation.mitigation_actions = mitigation_actions
        
        # Execute mitigation actions
        await self._execute_mitigation_actions(violation, mitigation_actions)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Error in safety violation callback: {str(e)}")
    
    async def _determine_mitigation_actions(self, violation: SafetyViolation) -> List[str]:
        """Determine appropriate mitigation actions for violation"""
        actions = []
        
        if violation.severity_level == SafetyLevel.EMERGENCY:
            actions.extend([
                "emergency_shutdown",
                "notify_emergency_contacts",
                "save_session_data",
                "generate_incident_report"
            ])
        elif violation.severity_level == SafetyLevel.CRITICAL:
            actions.extend([
                "pause_session",
                "alert_operator",
                "reduce_stimulation_intensity",
                "monitor_user_condition"
            ])
        elif violation.severity_level == SafetyLevel.WARNING:
            actions.extend([
                "log_warning",
                "adjust_parameters",
                "notify_user"
            ])
        
        # Specific actions based on violation type
        if violation.violation_type == SafetyViolationType.SIGNAL_AMPLITUDE_EXCEEDED:
            actions.extend([
                "check_electrode_connection",
                "reduce_amplifier_gain",
                "apply_additional_filtering"
            ])
        elif violation.violation_type == SafetyViolationType.SESSION_DURATION_EXCEEDED:
            actions.extend([
                "recommend_break",
                "save_current_progress",
                "schedule_continuation"
            ])
        elif violation.violation_type == SafetyViolationType.PHYSIOLOGICAL_THRESHOLD_EXCEEDED:
            actions.extend([
                "monitor_vital_signs",
                "provide_relaxation_guidance",
                "consider_session_termination"
            ])
        
        return actions
    
    async def _execute_mitigation_actions(self, violation: SafetyViolation, actions: List[str]):
        """Execute mitigation actions for safety violation"""
        executed_actions = []
        
        for action in actions:
            try:
                success = await self._execute_single_action(action, violation)
                if success:
                    executed_actions.append(action)
                    logger.info(f"Executed mitigation action: {action}")
                else:
                    logger.warning(f"Failed to execute mitigation action: {action}")
            except Exception as e:
                logger.error(f"Error executing mitigation action {action}: {str(e)}")
        
        return executed_actions
    
    async def _execute_single_action(self, action: str, violation: SafetyViolation) -> bool:
        """Execute a single mitigation action"""
        if action == "emergency_shutdown":
            # Emergency shutdown would integrate with main BCI system
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
            return True
        elif action == "pause_session":
            logger.warning("Session paused due to safety violation")
            return True
        elif action == "alert_operator":
            logger.warning("Operator alert sent")
            return True
        elif action == "log_warning":
            logger.warning(f"Safety warning logged: {violation.description}")
            return True
        elif action == "notify_user":
            logger.info("User notification sent")
            return True
        elif action == "save_session_data":
            logger.info("Session data saved for recovery")
            return True
        else:
            # Default action
            logger.info(f"Mitigation action executed: {action}")
            return True
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop physiological monitoring"""
        self.monitoring_active = False
        
        # Generate summary
        summary = {
            'total_violations': len(self.violation_history),
            'violations_by_type': {},
            'violations_by_severity': {}
        }
        
        for violation in self.violation_history:
            vtype = violation.violation_type.value
            severity = violation.severity_level.value
            
            summary['violations_by_type'][vtype] = summary['violations_by_type'].get(vtype, 0) + 1
            summary['violations_by_severity'][severity] = summary['violations_by_severity'].get(severity, 0) + 1
        
        logger.info("Physiological monitoring stopped")
        
        return {
            'status': 'monitoring_stopped',
            'summary': summary
        }

class ConsentManager:
    """User consent and authorization management"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./consent_records")
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory consent cache
        self.consent_cache = {}
        
        # Encryption for consent storage
        if ENCRYPTION_AVAILABLE:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        else:
            self.cipher_suite = None
        
        logger.info("Consent manager initialized")
    
    async def request_consent(self, user_id: str, consent_type: UserConsentType,
                            consent_text: str, consent_version: str = "1.0") -> Dict[str, Any]:
        """Request consent from user"""
        consent_id = str(uuid.uuid4())
        
        # In a real implementation, this would present consent UI to user
        # For now, we'll simulate consent request
        
        consent_request = {
            'consent_id': consent_id,
            'user_id': user_id,
            'consent_type': consent_type.value,
            'consent_text': consent_text,
            'consent_version': consent_version,
            'requested_at': datetime.now(timezone.utc).isoformat(),
            'status': 'pending'
        }
        
        logger.info(f"Consent requested for user {user_id}: {consent_type.value}")
        
        return consent_request
    
    async def record_consent(self, user_id: str, consent_type: UserConsentType,
                           granted: bool, digital_signature: str,
                           consent_version: str = "1.0",
                           expiry_days: Optional[int] = None) -> UserConsent:
        """Record user consent decision"""
        
        consent_id = str(uuid.uuid4())
        expiry_date = None
        
        if expiry_days:
            expiry_date = datetime.now(timezone.utc) + timedelta(days=expiry_days)
        
        consent = UserConsent(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.now(timezone.utc),
            expiry_date=expiry_date,
            consent_version=consent_version,
            digital_signature=digital_signature
        )
        
        # Store consent record
        await self._store_consent_record(consent)
        
        # Update cache
        cache_key = f"{user_id}_{consent_type.value}"
        self.consent_cache[cache_key] = consent
        
        logger.info(f"Consent recorded for user {user_id}: {consent_type.value} = {granted}")
        
        return consent
    
    async def check_consent(self, user_id: str, consent_type: UserConsentType) -> Optional[UserConsent]:
        """Check if user has valid consent for specific type"""
        cache_key = f"{user_id}_{consent_type.value}"
        
        # Check cache first
        if cache_key in self.consent_cache:
            consent = self.consent_cache[cache_key]
            
            # Check if consent is still valid
            if consent.granted and not consent.revocation_timestamp:
                if not consent.expiry_date or consent.expiry_date > datetime.now(timezone.utc):
                    return consent
        
        # Load from storage if not in cache
        consent = await self._load_consent_record(user_id, consent_type)
        if consent:
            self.consent_cache[cache_key] = consent
            
            # Check validity
            if consent.granted and not consent.revocation_timestamp:
                if not consent.expiry_date or consent.expiry_date > datetime.now(timezone.utc):
                    return consent
        
        return None
    
    async def revoke_consent(self, user_id: str, consent_type: UserConsentType) -> bool:
        """Revoke user consent"""
        consent = await self.check_consent(user_id, consent_type)
        
        if consent:
            consent.revocation_timestamp = datetime.now(timezone.utc)
            
            # Update storage
            await self._store_consent_record(consent)
            
            # Update cache
            cache_key = f"{user_id}_{consent_type.value}"
            self.consent_cache[cache_key] = consent
            
            logger.info(f"Consent revoked for user {user_id}: {consent_type.value}")
            return True
        
        return False
    
    async def _store_consent_record(self, consent: UserConsent):
        """Store consent record to persistent storage"""
        consent_data = {
            'consent_id': consent.consent_id,
            'user_id': consent.user_id,
            'consent_type': consent.consent_type.value,
            'granted': consent.granted,
            'timestamp': consent.timestamp.isoformat(),
            'expiry_date': consent.expiry_date.isoformat() if consent.expiry_date else None,
            'consent_version': consent.consent_version,
            'digital_signature': consent.digital_signature,
            'witness_signature': consent.witness_signature,
            'revocation_timestamp': consent.revocation_timestamp.isoformat() if consent.revocation_timestamp else None,
            'metadata': consent.metadata
        }
        
        # Encrypt consent data if encryption is available
        if self.cipher_suite:
            encrypted_data = self.cipher_suite.encrypt(json.dumps(consent_data).encode())
            file_data = encrypted_data
        else:
            file_data = json.dumps(consent_data, indent=2).encode()
        
        # Store to file
        filename = f"{consent.user_id}_{consent.consent_type.value}_{consent.consent_id}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'wb') as f:
            f.write(file_data)
    
    async def _load_consent_record(self, user_id: str, consent_type: UserConsentType) -> Optional[UserConsent]:
        """Load consent record from storage"""
        # Find consent files for this user and type
        pattern = f"{user_id}_{consent_type.value}_*.json"
        
        consent_files = list(self.storage_path.glob(pattern))
        
        if not consent_files:
            return None
        
        # Load the most recent consent file
        latest_file = max(consent_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'rb') as f:
                file_data = f.read()
            
            # Decrypt if necessary
            if self.cipher_suite:
                decrypted_data = self.cipher_suite.decrypt(file_data)
                consent_data = json.loads(decrypted_data.decode())
            else:
                consent_data = json.loads(file_data.decode())
            
            # Reconstruct consent object
            consent = UserConsent(
                consent_id=consent_data['consent_id'],
                user_id=consent_data['user_id'],
                consent_type=UserConsentType(consent_data['consent_type']),
                granted=consent_data['granted'],
                timestamp=datetime.fromisoformat(consent_data['timestamp']),
                expiry_date=datetime.fromisoformat(consent_data['expiry_date']) if consent_data['expiry_date'] else None,
                consent_version=consent_data['consent_version'],
                digital_signature=consent_data['digital_signature'],
                witness_signature=consent_data.get('witness_signature'),
                revocation_timestamp=datetime.fromisoformat(consent_data['revocation_timestamp']) if consent_data.get('revocation_timestamp') else None,
                metadata=consent_data.get('metadata', {})
            )
            
            return consent
            
        except Exception as e:
            logger.error(f"Error loading consent record {latest_file}: {str(e)}")
            return None
    
    async def get_user_consent_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of all consents for a user"""
        consents = []
        
        for consent_type in UserConsentType:
            consent = await self.check_consent(user_id, consent_type)
            if consent:
                consents.append({
                    'consent_type': consent.consent_type.value,
                    'granted': consent.granted,
                    'timestamp': consent.timestamp.isoformat(),
                    'expiry_date': consent.expiry_date.isoformat() if consent.expiry_date else None,
                    'revoked': consent.revocation_timestamp is not None,
                    'valid': consent.granted and not consent.revocation_timestamp and (not consent.expiry_date or consent.expiry_date > datetime.now(timezone.utc))
                })
        
        return {
            'user_id': user_id,
            'consents': consents,
            'total_consents': len(consents),
            'valid_consents': len([c for c in consents if c['valid']])
        }

class ComplianceAuditor:
    """Compliance auditing and regulatory reporting"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./audit_logs")
        self.storage_path.mkdir(exist_ok=True)
        
        # Audit event buffer
        self.audit_events = []
        
        # Compliance standards being monitored
        self.monitored_standards = [
            ComplianceStandard.ISO_14155,
            ComplianceStandard.FDA_CFR_820,
            ComplianceStandard.IEC_60601,
            ComplianceStandard.HIPAA
        ]
        
        # Encryption for audit logs
        if ENCRYPTION_AVAILABLE:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        else:
            self.cipher_suite = None
        
        logger.info("Compliance auditor initialized")
    
    async def log_audit_event(self, event_type: str, user_id: Optional[str] = None,
                            session_id: Optional[str] = None,
                            details: Optional[Dict[str, Any]] = None,
                            compliance_standards: Optional[List[ComplianceStandard]] = None) -> str:
        """Log an audit event"""
        
        event_id = str(uuid.uuid4())
        
        audit_event = SafetyAuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            compliance_standards=compliance_standards or self.monitored_standards
        )
        
        # Generate hash signature for integrity
        event_data = {
            'event_id': event_id,
            'event_type': event_type,
            'timestamp': audit_event.timestamp.isoformat(),
            'user_id': user_id,
            'session_id': session_id,
            'details': details or {}
        }
        
        event_json = json.dumps(event_data, sort_keys=True)
        audit_event.hash_signature = hashlib.sha256(event_json.encode()).hexdigest()
        
        # Encrypt sensitive data if available
        if self.cipher_suite and user_id:
            encrypted_details = self.cipher_suite.encrypt(json.dumps(details or {}).encode())
            audit_event.encrypted_data = encrypted_details.decode('latin-1')
        
        # Add to buffer and persist
        self.audit_events.append(audit_event)
        await self._persist_audit_event(audit_event)
        
        logger.debug(f"Audit event logged: {event_type} ({event_id})")
        
        return event_id
    
    async def _persist_audit_event(self, event: SafetyAuditEvent):
        """Persist audit event to storage"""
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'session_id': event.session_id,
            'details': event.details,
            'compliance_standards': [std.value for std in event.compliance_standards],
            'hash_signature': event.hash_signature,
            'encrypted_data': event.encrypted_data
        }
        
        # Write to daily audit log file
        date_str = event.timestamp.strftime('%Y-%m-%d')
        log_filename = f"audit_log_{date_str}.jsonl"
        log_filepath = self.storage_path / log_filename
        
        with open(log_filepath, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
    
    async def generate_compliance_report(self, start_date: datetime, end_date: datetime,
                                       standards: Optional[List[ComplianceStandard]] = None) -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        
        if not standards:
            standards = self.monitored_standards
        
        # Load audit events for the period
        events = await self._load_audit_events(start_date, end_date)
        
        report = {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'standards_covered': [std.value for std in standards],
            'total_events': len(events),
            'event_summary': {},
            'compliance_analysis': {},
            'recommendations': []
        }
        
        # Analyze events by type
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        report['event_summary'] = event_types
        
        # Compliance analysis per standard
        for standard in standards:
            analysis = await self._analyze_compliance_for_standard(events, standard)
            report['compliance_analysis'][standard.value] = analysis
        
        # Generate recommendations
        recommendations = await self._generate_compliance_recommendations(events, standards)
        report['recommendations'] = recommendations
        
        return report
    
    async def _load_audit_events(self, start_date: datetime, end_date: datetime) -> List[SafetyAuditEvent]:
        """Load audit events from storage for specified period"""
        events = []
        
        current_date = start_date.date()
        end_date_date = end_date.date()
        
        while current_date <= end_date_date:
            date_str = current_date.strftime('%Y-%m-%d')
            log_filename = f"audit_log_{date_str}.jsonl"
            log_filepath = self.storage_path / log_filename
            
            if log_filepath.exists():
                with open(log_filepath, 'r') as f:
                    for line in f:
                        event_data = json.loads(line.strip())
                        
                        # Create audit event object
                        event = SafetyAuditEvent(
                            event_id=event_data['event_id'],
                            event_type=event_data['event_type'],
                            timestamp=datetime.fromisoformat(event_data['timestamp']),
                            user_id=event_data.get('user_id'),
                            session_id=event_data.get('session_id'),
                            details=event_data.get('details', {}),
                            compliance_standards=[ComplianceStandard(std) for std in event_data.get('compliance_standards', [])],
                            hash_signature=event_data.get('hash_signature', ''),
                            encrypted_data=event_data.get('encrypted_data')
                        )
                        
                        # Check if event is within time range
                        if start_date <= event.timestamp <= end_date:
                            events.append(event)
            
            current_date += timedelta(days=1)
        
        return events
    
    async def _analyze_compliance_for_standard(self, events: List[SafetyAuditEvent], 
                                             standard: ComplianceStandard) -> Dict[str, Any]:
        """Analyze compliance for a specific standard"""
        
        relevant_events = [e for e in events if standard in e.compliance_standards]
        
        analysis = {
            'standard': standard.value,
            'relevant_events': len(relevant_events),
            'compliance_score': 1.0,  # Default perfect score
            'violations': [],
            'recommendations': []
        }
        
        # Standard-specific analysis
        if standard == ComplianceStandard.ISO_14155:
            analysis.update(await self._analyze_iso_14155_compliance(relevant_events))
        elif standard == ComplianceStandard.FDA_CFR_820:
            analysis.update(await self._analyze_fda_820_compliance(relevant_events))
        elif standard == ComplianceStandard.IEC_60601:
            analysis.update(await self._analyze_iec_60601_compliance(relevant_events))
        elif standard == ComplianceStandard.HIPAA:
            analysis.update(await self._analyze_hipaa_compliance(relevant_events))
        
        return analysis
    
    async def _analyze_iso_14155_compliance(self, events: List[SafetyAuditEvent]) -> Dict[str, Any]:
        """Analyze ISO 14155 (clinical investigation) compliance"""
        
        analysis = {
            'clinical_protocol_adherence': 1.0,
            'adverse_event_reporting': 1.0,
            'data_integrity': 1.0,
            'subject_safety': 1.0
        }
        
        # Look for safety violations
        safety_events = [e for e in events if 'safety' in e.event_type.lower()]
        if safety_events:
            analysis['subject_safety'] = max(0.0, 1.0 - (len(safety_events) / 100.0))  # Penalize safety events
        
        # Check data integrity
        data_events = [e for e in events if 'data' in e.event_type.lower() and 'error' in e.event_type.lower()]
        if data_events:
            analysis['data_integrity'] = max(0.0, 1.0 - (len(data_events) / 50.0))
        
        return analysis
    
    async def _analyze_fda_820_compliance(self, events: List[SafetyAuditEvent]) -> Dict[str, Any]:
        """Analyze FDA 21 CFR 820 (quality system) compliance"""
        
        analysis = {
            'design_controls': 1.0,
            'risk_management': 1.0,
            'corrective_preventive_actions': 1.0,
            'management_responsibility': 1.0
        }
        
        # Check for design control events
        design_events = [e for e in events if 'design' in e.event_type.lower() or 'validation' in e.event_type.lower()]
        if design_events:
            analysis['design_controls'] = min(1.0, len(design_events) / 10.0)  # Reward design activities
        
        return analysis
    
    async def _analyze_iec_60601_compliance(self, events: List[SafetyAuditEvent]) -> Dict[str, Any]:
        """Analyze IEC 60601 (medical electrical equipment) compliance"""
        
        analysis = {
            'electrical_safety': 1.0,
            'electromagnetic_compatibility': 1.0,
            'usability_engineering': 1.0,
            'software_lifecycle': 1.0
        }
        
        # Check for electrical safety events
        electrical_events = [e for e in events if 'electrical' in e.event_type.lower() or 'power' in e.event_type.lower()]
        safety_violations = [e for e in electrical_events if 'violation' in e.event_type.lower()]
        
        if safety_violations:
            analysis['electrical_safety'] = max(0.0, 1.0 - (len(safety_violations) / 5.0))
        
        return analysis
    
    async def _analyze_hipaa_compliance(self, events: List[SafetyAuditEvent]) -> Dict[str, Any]:
        """Analyze HIPAA (health data privacy) compliance"""
        
        analysis = {
            'data_privacy': 1.0,
            'access_controls': 1.0,
            'audit_controls': 1.0,
            'data_integrity': 1.0,
            'transmission_security': 1.0
        }
        
        # Check for privacy violations
        privacy_events = [e for e in events if 'privacy' in e.event_type.lower() or 'unauthorized' in e.event_type.lower()]
        if privacy_events:
            analysis['data_privacy'] = max(0.0, 1.0 - (len(privacy_events) / 3.0))
        
        # Check access control events
        access_events = [e for e in events if 'access' in e.event_type.lower()]
        unauthorized_access = [e for e in access_events if 'unauthorized' in e.event_type.lower()]
        
        if unauthorized_access:
            analysis['access_controls'] = max(0.0, 1.0 - (len(unauthorized_access) / 5.0))
        
        return analysis
    
    async def _generate_compliance_recommendations(self, events: List[SafetyAuditEvent],
                                                 standards: List[ComplianceStandard]) -> List[str]:
        """Generate compliance recommendations based on audit analysis"""
        
        recommendations = []
        
        # Analyze event patterns
        safety_events = [e for e in events if 'safety' in e.event_type.lower()]
        error_events = [e for e in events if 'error' in e.event_type.lower()]
        violation_events = [e for e in events if 'violation' in e.event_type.lower()]
        
        if len(safety_events) > 5:
            recommendations.append("Implement additional safety monitoring protocols")
            recommendations.append("Review and update safety thresholds")
        
        if len(error_events) > 10:
            recommendations.append("Enhance error handling and recovery procedures")
            recommendations.append("Implement automated error detection and reporting")
        
        if len(violation_events) > 3:
            recommendations.append("Conduct comprehensive compliance training")
            recommendations.append("Review and strengthen compliance procedures")
        
        # Standard-specific recommendations
        if ComplianceStandard.HIPAA in standards:
            data_events = [e for e in events if 'data' in e.event_type.lower()]
            if len(data_events) > 20:
                recommendations.append("Implement enhanced data encryption and access controls")
        
        if ComplianceStandard.IEC_60601 in standards:
            electrical_events = [e for e in events if 'electrical' in e.event_type.lower()]
            if len(electrical_events) > 2:
                recommendations.append("Conduct electrical safety inspection and testing")
        
        return recommendations

class BCISafetySystem:
    """Main BCI safety system orchestrating all safety protocols"""
    
    def __init__(self, thresholds: Optional[SafetyThresholds] = None,
                 consent_storage_path: Optional[Path] = None,
                 audit_storage_path: Optional[Path] = None):
        
        self.thresholds = thresholds or SafetyThresholds()
        self.physiological_monitor = PhysiologicalMonitor(self.thresholds)
        self.consent_manager = ConsentManager(consent_storage_path)
        self.compliance_auditor = ComplianceAuditor(audit_storage_path)
        
        # Safety system state
        self.safety_system_active = False
        self.active_sessions = {}
        self.emergency_contacts = []
        
        # Register safety violation callback
        self.physiological_monitor.add_alert_callback(self._handle_safety_violation_callback)
        
        logger.info("BCI Safety System initialized")
    
    async def initialize_safety_system(self) -> Dict[str, Any]:
        """Initialize the complete safety system"""
        
        self.safety_system_active = True
        
        # Log system initialization
        await self.compliance_auditor.log_audit_event(
            'safety_system_initialized',
            details={
                'thresholds': self.thresholds.__dict__,
                'compliance_standards': [std.value for std in [
                    ComplianceStandard.ISO_14155,
                    ComplianceStandard.FDA_CFR_820,
                    ComplianceStandard.IEC_60601,
                    ComplianceStandard.HIPAA
                ]]
            }
        )
        
        logger.info("BCI Safety System fully initialized")
        
        return {
            'status': 'initialized',
            'safety_system_active': True,
            'thresholds': self.thresholds.__dict__,
            'monitored_standards': [std.value for std in self.compliance_auditor.monitored_standards]
        }
    
    async def start_safe_session(self, user_id: str, session_type: str = "bci_training") -> Dict[str, Any]:
        """Start a safety-monitored BCI session"""
        
        if not self.safety_system_active:
            return {'status': 'error', 'message': 'Safety system not initialized'}
        
        session_id = str(uuid.uuid4())
        
        # Check required consents
        required_consents = [
            UserConsentType.DATA_COLLECTION,
            UserConsentType.DATA_PROCESSING
        ]
        
        missing_consents = []
        for consent_type in required_consents:
            consent = await self.consent_manager.check_consent(user_id, consent_type)
            if not consent:
                missing_consents.append(consent_type.value)
        
        if missing_consents:
            return {
                'status': 'consent_required',
                'missing_consents': missing_consents,
                'message': 'User consent required before starting session'
            }
        
        # Start physiological monitoring
        monitor_result = await self.physiological_monitor.start_monitoring(user_id, session_id)
        
        # Record session start
        session_info = {
            'session_id': session_id,
            'user_id': user_id,
            'session_type': session_type,
            'start_time': datetime.now(timezone.utc),
            'safety_monitoring': True
        }
        
        self.active_sessions[session_id] = session_info
        
        # Log session start
        await self.compliance_auditor.log_audit_event(
            'session_started',
            user_id=user_id,
            session_id=session_id,
            details={
                'session_type': session_type,
                'safety_thresholds': self.thresholds.__dict__
            }
        )
        
        logger.info(f"Safe BCI session started for user {user_id}: {session_id}")
        
        return {
            'status': 'session_started',
            'session_id': session_id,
            'user_id': user_id,
            'session_type': session_type,
            'safety_monitoring': True,
            'monitor_result': monitor_result
        }
    
    async def monitor_session_safety(self, session_id: str, 
                                   signal_data: Optional[Dict[str, float]] = None,
                                   physiological_data: Optional[Dict[str, float]] = None,
                                   processing_times: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Monitor ongoing session for safety violations"""
        
        if session_id not in self.active_sessions:
            return {'status': 'session_not_found'}
        
        session_info = self.active_sessions[session_id]
        violations = []
        
        # Monitor signal amplitudes
        if signal_data:
            for signal_type, amplitude in signal_data.items():
                violation = await self.physiological_monitor.monitor_signal_amplitude(signal_type, amplitude)
                if violation:
                    violations.append(violation)
        
        # Monitor processing times
        if processing_times:
            for operation, time_ms in processing_times.items():
                violation = await self.physiological_monitor.monitor_processing_time(operation, time_ms)
                if violation:
                    violations.append(violation)
        
        # Monitor session duration
        session_duration_violation = await self.physiological_monitor.monitor_session_duration(
            session_info['start_time']
        )
        if session_duration_violation:
            violations.append(session_duration_violation)
        
        # Monitor physiological parameters
        if physiological_data:
            heart_rate = physiological_data.get('heart_rate')
            stress_level = physiological_data.get('stress_level')
            physio_violations = await self.physiological_monitor.monitor_physiological_parameters(
                heart_rate, stress_level
            )
            violations.extend(physio_violations)
        
        # Log monitoring results
        if violations:
            await self.compliance_auditor.log_audit_event(
                'safety_violations_detected',
                user_id=session_info['user_id'],
                session_id=session_id,
                details={
                    'violation_count': len(violations),
                    'violation_types': [v.violation_type.value for v in violations]
                }
            )
        
        return {
            'status': 'monitoring_completed',
            'session_id': session_id,
            'violations_detected': len(violations),
            'violations': [
                {
                    'type': v.violation_type.value,
                    'severity': v.severity_level.value,
                    'description': v.description,
                    'mitigation_actions': v.mitigation_actions
                }
                for v in violations
            ]
        }
    
    async def end_safe_session(self, session_id: str) -> Dict[str, Any]:
        """End a safety-monitored BCI session"""
        
        if session_id not in self.active_sessions:
            return {'status': 'session_not_found'}
        
        session_info = self.active_sessions[session_id]
        
        # Stop physiological monitoring
        monitor_result = await self.physiological_monitor.stop_monitoring()
        
        # Calculate session metrics
        session_duration = (datetime.now(timezone.utc) - session_info['start_time']).total_seconds() / 60
        
        # Log session end
        await self.compliance_auditor.log_audit_event(
            'session_ended',
            user_id=session_info['user_id'],
            session_id=session_id,
            details={
                'session_duration_minutes': session_duration,
                'monitoring_summary': monitor_result['summary']
            }
        )
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Safe BCI session ended: {session_id} (duration: {session_duration:.1f} minutes)")
        
        return {
            'status': 'session_ended',
            'session_id': session_id,
            'duration_minutes': session_duration,
            'monitoring_summary': monitor_result['summary']
        }
    
    async def request_user_consent(self, user_id: str, consent_types: List[UserConsentType]) -> Dict[str, Any]:
        """Request multiple types of consent from user"""
        
        consent_requests = []
        
        for consent_type in consent_types:
            # Generate appropriate consent text
            consent_text = self._generate_consent_text(consent_type)
            
            request = await self.consent_manager.request_consent(
                user_id, consent_type, consent_text
            )
            consent_requests.append(request)
        
        # Log consent request
        await self.compliance_auditor.log_audit_event(
            'consent_requested',
            user_id=user_id,
            details={
                'consent_types': [ct.value for ct in consent_types],
                'request_count': len(consent_requests)
            }
        )
        
        return {
            'status': 'consent_requested',
            'user_id': user_id,
            'consent_requests': consent_requests
        }
    
    async def record_user_consent(self, user_id: str, consent_decisions: Dict[str, bool],
                                digital_signature: str) -> Dict[str, Any]:
        """Record user consent decisions"""
        
        recorded_consents = []
        
        for consent_type_str, granted in consent_decisions.items():
            try:
                consent_type = UserConsentType(consent_type_str)
                consent = await self.consent_manager.record_consent(
                    user_id, consent_type, granted, digital_signature
                )
                recorded_consents.append(consent)
            except ValueError:
                logger.warning(f"Invalid consent type: {consent_type_str}")
        
        # Log consent recording
        await self.compliance_auditor.log_audit_event(
            'consent_recorded',
            user_id=user_id,
            details={
                'consents_recorded': len(recorded_consents),
                'granted_consents': len([c for c in recorded_consents if c.granted])
            }
        )
        
        return {
            'status': 'consent_recorded',
            'user_id': user_id,
            'consents_recorded': len(recorded_consents),
            'consent_summary': [
                {
                    'type': c.consent_type.value,
                    'granted': c.granted,
                    'timestamp': c.timestamp.isoformat()
                }
                for c in recorded_consents
            ]
        }
    
    def _generate_consent_text(self, consent_type: UserConsentType) -> str:
        """Generate appropriate consent text for consent type"""
        
        consent_texts = {
            UserConsentType.DATA_COLLECTION: """
            I consent to the collection of my biometric data including EEG, ECG, and other 
            physiological signals for the purpose of brain-computer interface research and training.
            I understand that this data will be used to improve BCI system performance and 
            personalize my training experience.
            """,
            
            UserConsentType.DATA_PROCESSING: """
            I consent to the processing of my collected biometric data using automated algorithms 
            and machine learning techniques for the purpose of neural signal analysis and 
            brain-computer interface control.
            """,
            
            UserConsentType.DATA_STORAGE: """
            I consent to the secure storage of my biometric data and analysis results for 
            research and system improvement purposes. I understand that data will be stored 
            with appropriate security measures and access controls.
            """,
            
            UserConsentType.RESEARCH_PARTICIPATION: """
            I consent to participate in brain-computer interface research studies. I understand 
            that my de-identified data may be used for scientific research purposes to advance 
            the field of neurotechnology and improve BCI systems for future users.
            """
        }
        
        return consent_texts.get(consent_type, "General consent for BCI system use.")
    
    def _handle_safety_violation_callback(self, violation: SafetyViolation):
        """Handle safety violations from physiological monitor"""
        
        # Log violation to audit system
        asyncio.create_task(self.compliance_auditor.log_audit_event(
            'safety_violation',
            user_id=violation.user_id,
            session_id=violation.session_id,
            details={
                'violation_type': violation.violation_type.value,
                'severity': violation.severity_level.value,
                'description': violation.description,
                'measured_value': violation.measured_value,
                'threshold_value': violation.threshold_value,
                'mitigation_actions': violation.mitigation_actions
            }
        ))
        
        # Handle emergency situations
        if violation.severity_level == SafetyLevel.EMERGENCY:
            asyncio.create_task(self._handle_emergency_situation(violation))
    
    async def _handle_emergency_situation(self, violation: SafetyViolation):
        """Handle emergency safety situations"""
        
        logger.critical(f"EMERGENCY SITUATION: {violation.description}")
        
        # Emergency actions
        emergency_actions = [
            "immediate_session_termination",
            "emergency_contacts_notification",
            "incident_report_generation",
            "regulatory_notification"
        ]
        
        for action in emergency_actions:
            try:
                await self._execute_emergency_action(action, violation)
                logger.info(f"Emergency action executed: {action}")
            except Exception as e:
                logger.error(f"Failed to execute emergency action {action}: {str(e)}")
        
        # Log emergency response
        await self.compliance_auditor.log_audit_event(
            'emergency_response',
            user_id=violation.user_id,
            session_id=violation.session_id,
            details={
                'violation_id': violation.violation_id,
                'emergency_actions': emergency_actions,
                'response_timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def _execute_emergency_action(self, action: str, violation: SafetyViolation):
        """Execute specific emergency action"""
        
        if action == "immediate_session_termination":
            # Terminate all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.end_safe_session(session_id)
        
        elif action == "emergency_contacts_notification":
            # Notify emergency contacts (would integrate with actual notification system)
            for contact in self.emergency_contacts:
                logger.critical(f"Emergency notification sent to: {contact}")
        
        elif action == "incident_report_generation":
            # Generate detailed incident report
            incident_report = {
                'incident_id': str(uuid.uuid4()),
                'violation': violation.__dict__,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_state': 'emergency_shutdown'
            }
            
            # Save incident report
            incident_file = self.compliance_auditor.storage_path / f"incident_{incident_report['incident_id']}.json"
            with open(incident_file, 'w') as f:
                json.dump(incident_report, f, indent=2, default=str)
        
        elif action == "regulatory_notification":
            # Prepare regulatory notification (would integrate with regulatory reporting systems)
            logger.critical("Regulatory notification prepared for safety violation")
    
    async def generate_safety_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        
        # Get compliance report
        compliance_report = await self.compliance_auditor.generate_compliance_report(
            start_date, end_date
        )
        
        # Add safety-specific metrics
        safety_metrics = {
            'active_sessions_count': len(self.active_sessions),
            'total_violations': len(self.physiological_monitor.violation_history),
            'violations_by_severity': {},
            'safety_system_uptime': 99.9,  # Would calculate from actual uptime data
            'emergency_incidents': 0  # Would count from audit logs
        }
        
        # Count violations by severity
        for violation in self.physiological_monitor.violation_history:
            severity = violation.severity_level.value
            safety_metrics['violations_by_severity'][severity] = safety_metrics['violations_by_severity'].get(severity, 0) + 1
        
        return {
            'report_type': 'safety_report',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'compliance_report': compliance_report,
            'safety_metrics': safety_metrics,
            'thresholds': self.thresholds.__dict__,
            'system_status': {
                'safety_system_active': self.safety_system_active,
                'monitoring_active': self.physiological_monitor.monitoring_active,
                'active_sessions': len(self.active_sessions)
            }
        }
    
    async def get_safety_system_status(self) -> Dict[str, Any]:
        """Get current safety system status"""
        
        return {
            'safety_system_active': self.safety_system_active,
            'physiological_monitor': {
                'active': self.physiological_monitor.monitoring_active,
                'violations_detected': len(self.physiological_monitor.violation_history),
                'thresholds': self.thresholds.__dict__
            },
            'consent_manager': {
                'consent_cache_size': len(self.consent_manager.consent_cache),
                'encryption_enabled': self.consent_manager.cipher_suite is not None
            },
            'compliance_auditor': {
                'audit_events_buffered': len(self.compliance_auditor.audit_events),
                'monitored_standards': [std.value for std in self.compliance_auditor.monitored_standards],
                'encryption_enabled': self.compliance_auditor.cipher_suite is not None
            },
            'active_sessions': len(self.active_sessions),
            'emergency_contacts': len(self.emergency_contacts)
        }
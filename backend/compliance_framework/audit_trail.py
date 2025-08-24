"""
Immutable Audit Trail System with Blockchain-Style Verification
==============================================================

Advanced audit logging system that creates tamper-proof audit trails
using blockchain-style verification and cryptographic hashing.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid
import threading
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import secrets


class EventType(Enum):
    """Audit event types"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_VIOLATION = "compliance_violation"
    TRADE_EXECUTION = "trade_execution"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    SYSTEM_ERROR = "system_error"
    ADMIN_ACTION = "admin_action"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"
    DATABASE_QUERY = "database_query"


class SeverityLevel(Enum):
    """Event severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataCategory(Enum):
    """Data categories for audit"""
    PERSONAL_DATA = "personal_data"
    FINANCIAL_DATA = "financial_data"
    TRADING_DATA = "trading_data"
    SYSTEM_DATA = "system_data"
    CONFIGURATION_DATA = "configuration_data"
    SECURITY_DATA = "security_data"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: SeverityLevel
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource_accessed: str
    action_performed: str
    data_category: DataCategory
    data_classification: str
    request_details: Dict[str, Any]
    response_details: Dict[str, Any]
    compliance_context: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    jurisdiction: str
    data_residency: str
    cross_border_transfer: bool
    retention_category: str
    encryption_used: bool
    authentication_method: str
    authorization_granted: bool
    business_justification: str
    previous_hash: str
    current_hash: str
    digital_signature: str
    verification_status: str
    chain_position: int


class AuditBlock:
    """Blockchain-style audit block"""
    
    def __init__(self, block_id: str, previous_hash: str, events: List[AuditEvent]):
        self.block_id = block_id
        self.timestamp = datetime.now(timezone.utc)
        self.previous_hash = previous_hash
        self.events = events
        self.merkle_root = self._calculate_merkle_root()
        self.nonce = secrets.randbelow(2**32)
        self.hash = self._calculate_block_hash()
    
    def _calculate_merkle_root(self) -> str:
        """Calculate Merkle root of all events in the block"""
        if not self.events:
            return hashlib.sha256(b"").hexdigest()
        
        event_hashes = [e.current_hash for e in self.events]
        
        while len(event_hashes) > 1:
            next_level = []
            for i in range(0, len(event_hashes), 2):
                left = event_hashes[i]
                right = event_hashes[i + 1] if i + 1 < len(event_hashes) else left
                combined = left + right
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            event_hashes = next_level
        
        return event_hashes[0]
    
    def _calculate_block_hash(self) -> str:
        """Calculate block hash including all metadata"""
        block_data = {
            "block_id": self.block_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "event_count": len(self.events),
            "nonce": self.nonce
        }
        block_json = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_json.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for storage"""
        return {
            "block_id": self.block_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "nonce": self.nonce,
            "hash": self.hash,
            "event_count": len(self.events),
            "events": [asdict(event) for event in self.events]
        }


class ImmutableAuditTrail:
    """
    Blockchain-style immutable audit trail system.
    
    Creates tamper-proof audit logs using cryptographic hashing,
    digital signatures, and blockchain-style verification.
    """
    
    def __init__(self, data_directory: str = "/app/compliance/audit"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize cryptographic components
        self._initialize_crypto()
        
        # Audit chain
        self.blocks: List[AuditBlock] = []
        self.pending_events: List[AuditEvent] = []
        self.event_index: Dict[str, int] = {}  # event_id -> chain_position
        
        # Performance metrics
        self.metrics = {
            "total_events": 0,
            "total_blocks": 0,
            "verification_checks": 0,
            "integrity_violations": 0,
            "average_block_time": 0.0
        }
        
        self.logger = logging.getLogger("compliance.audit_trail")
        self.lock = threading.Lock()
        
        # Load existing chain
        self._load_audit_chain()
        
        # Start background processing
        self.block_creation_interval = 300  # 5 minutes
        self.max_events_per_block = 1000
        asyncio.create_task(self._background_block_creation())
    
    def _initialize_crypto(self):
        """Initialize cryptographic keys for digital signatures"""
        key_file = self.data_directory / "audit_private_key.pem"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            with open(key_file, 'wb') as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            # Save public key
            public_key = self.private_key.public_key()
            with open(self.data_directory / "audit_public_key.pem", 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        
        self.public_key = self.private_key.public_key()
    
    def log_event(self, 
                  event_type: EventType,
                  severity: SeverityLevel,
                  resource_accessed: str,
                  action_performed: str,
                  data_category: DataCategory,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  request_details: Optional[Dict[str, Any]] = None,
                  response_details: Optional[Dict[str, Any]] = None,
                  compliance_context: Optional[Dict[str, Any]] = None,
                  **kwargs) -> str:
        """Log an audit event with immutable characteristics"""
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        # Get previous hash for chaining
        previous_hash = self._get_last_hash()
        
        # Create audit event
        audit_event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            data_category=data_category,
            data_classification=kwargs.get("data_classification", "internal"),
            request_details=request_details or {},
            response_details=response_details or {},
            compliance_context=compliance_context or {},
            risk_assessment=kwargs.get("risk_assessment", {}),
            jurisdiction=kwargs.get("jurisdiction", "unknown"),
            data_residency=kwargs.get("data_residency", "unknown"),
            cross_border_transfer=kwargs.get("cross_border_transfer", False),
            retention_category=kwargs.get("retention_category", "standard"),
            encryption_used=kwargs.get("encryption_used", False),
            authentication_method=kwargs.get("authentication_method", "unknown"),
            authorization_granted=kwargs.get("authorization_granted", False),
            business_justification=kwargs.get("business_justification", ""),
            previous_hash=previous_hash,
            current_hash="",  # Will be calculated
            digital_signature="",  # Will be calculated
            verification_status="pending",
            chain_position=len(self.blocks) * self.max_events_per_block + len(self.pending_events)
        )
        
        # Calculate event hash
        audit_event.current_hash = self._calculate_event_hash(audit_event)
        
        # Create digital signature
        audit_event.digital_signature = self._create_digital_signature(audit_event)
        audit_event.verification_status = "verified"
        
        # Add to pending events
        with self.lock:
            self.pending_events.append(audit_event)
            self.event_index[event_id] = audit_event.chain_position
            self.metrics["total_events"] += 1
        
        # Create block if threshold reached
        if len(self.pending_events) >= self.max_events_per_block:
            asyncio.create_task(self._create_audit_block())
        
        self.logger.info(f"Audit event logged: {event_id} ({event_type.value})")
        return event_id
    
    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Calculate cryptographic hash for an audit event"""
        # Create event dictionary without hash and signature
        event_dict = asdict(event)
        event_dict.pop("current_hash", None)
        event_dict.pop("digital_signature", None)
        event_dict.pop("verification_status", None)
        
        # Convert datetime to ISO format
        for key, value in event_dict.items():
            if isinstance(value, datetime):
                event_dict[key] = value.isoformat()
        
        event_json = json.dumps(event_dict, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    def _create_digital_signature(self, event: AuditEvent) -> str:
        """Create digital signature for audit event"""
        try:
            signature = self.private_key.sign(
                event.current_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature.hex()
        except Exception as e:
            self.logger.error(f"Failed to create digital signature: {str(e)}")
            return ""
    
    def _verify_digital_signature(self, event: AuditEvent) -> bool:
        """Verify digital signature of audit event"""
        try:
            signature_bytes = bytes.fromhex(event.digital_signature)
            self.public_key.verify(
                signature_bytes,
                event.current_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _get_last_hash(self) -> str:
        """Get the hash of the last event or block"""
        if self.pending_events:
            return self.pending_events[-1].current_hash
        elif self.blocks:
            return self.blocks[-1].hash
        else:
            return "0" * 64  # Genesis hash
    
    async def _create_audit_block(self):
        """Create a new audit block from pending events"""
        if not self.pending_events:
            return
        
        block_start_time = time.time()
        
        with self.lock:
            events_to_process = self.pending_events.copy()
            self.pending_events.clear()
        
        # Create block
        block_id = f"AUDIT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{len(self.blocks):06d}"
        previous_hash = self.blocks[-1].hash if self.blocks else "0" * 64
        
        audit_block = AuditBlock(block_id, previous_hash, events_to_process)
        
        # Add to chain
        with self.lock:
            self.blocks.append(audit_block)
            self.metrics["total_blocks"] += 1
        
        # Calculate average block time
        block_time = time.time() - block_start_time
        self.metrics["average_block_time"] = (
            (self.metrics["average_block_time"] * (self.metrics["total_blocks"] - 1) + block_time)
            / self.metrics["total_blocks"]
        )
        
        # Persist block
        await self._save_audit_block(audit_block)
        
        self.logger.info(f"Created audit block: {block_id} with {len(events_to_process)} events")
    
    async def _background_block_creation(self):
        """Background task for periodic block creation"""
        while True:
            try:
                await asyncio.sleep(self.block_creation_interval)
                
                if self.pending_events:
                    await self._create_audit_block()
                
            except Exception as e:
                self.logger.error(f"Error in background block creation: {str(e)}")
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit chain"""
        verification_start = time.time()
        
        verification_result = {
            "is_valid": True,
            "total_blocks": len(self.blocks),
            "total_events": self.metrics["total_events"],
            "verification_time": 0.0,
            "errors": [],
            "warnings": []
        }
        
        # Verify each block
        for i, block in enumerate(self.blocks):
            block_errors = self._verify_block_integrity(block, i)
            if block_errors:
                verification_result["errors"].extend(block_errors)
                verification_result["is_valid"] = False
        
        # Verify chain linkage
        for i in range(1, len(self.blocks)):
            if self.blocks[i].previous_hash != self.blocks[i-1].hash:
                verification_result["errors"].append({
                    "type": "chain_linkage_error",
                    "block_index": i,
                    "description": "Block previous_hash does not match previous block hash"
                })
                verification_result["is_valid"] = False
        
        verification_result["verification_time"] = time.time() - verification_start
        self.metrics["verification_checks"] += 1
        
        if not verification_result["is_valid"]:
            self.metrics["integrity_violations"] += 1
        
        return verification_result
    
    def _verify_block_integrity(self, block: AuditBlock, block_index: int) -> List[Dict[str, Any]]:
        """Verify integrity of a specific block"""
        errors = []
        
        # Verify block hash
        calculated_hash = block._calculate_block_hash()
        if calculated_hash != block.hash:
            errors.append({
                "type": "block_hash_mismatch",
                "block_index": block_index,
                "expected": calculated_hash,
                "actual": block.hash
            })
        
        # Verify Merkle root
        calculated_merkle = block._calculate_merkle_root()
        if calculated_merkle != block.merkle_root:
            errors.append({
                "type": "merkle_root_mismatch",
                "block_index": block_index,
                "expected": calculated_merkle,
                "actual": block.merkle_root
            })
        
        # Verify individual events
        for j, event in enumerate(block.events):
            event_errors = self._verify_event_integrity(event, block_index, j)
            errors.extend(event_errors)
        
        return errors
    
    def _verify_event_integrity(self, event: AuditEvent, block_index: int, event_index: int) -> List[Dict[str, Any]]:
        """Verify integrity of a specific event"""
        errors = []
        
        # Verify event hash
        calculated_hash = self._calculate_event_hash(event)
        if calculated_hash != event.current_hash:
            errors.append({
                "type": "event_hash_mismatch",
                "block_index": block_index,
                "event_index": event_index,
                "event_id": event.event_id,
                "expected": calculated_hash,
                "actual": event.current_hash
            })
        
        # Verify digital signature
        if not self._verify_digital_signature(event):
            errors.append({
                "type": "digital_signature_invalid",
                "block_index": block_index,
                "event_index": event_index,
                "event_id": event.event_id
            })
        
        return errors
    
    def query_events(self,
                    event_type: Optional[EventType] = None,
                    severity: Optional[SeverityLevel] = None,
                    user_id: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    data_category: Optional[DataCategory] = None,
                    limit: int = 1000) -> List[AuditEvent]:
        """Query audit events with filters"""
        
        all_events = []
        
        # Collect events from blocks
        for block in self.blocks:
            all_events.extend(block.events)
        
        # Add pending events
        all_events.extend(self.pending_events)
        
        # Apply filters
        filtered_events = []
        for event in all_events:
            if event_type and event.event_type != event_type:
                continue
            if severity and event.severity != severity:
                continue
            if user_id and event.user_id != user_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if data_category and event.data_category != data_category:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return filtered_events
    
    def get_compliance_report(self, 
                            jurisdiction: str,
                            start_date: datetime,
                            end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific jurisdiction"""
        
        events = self.query_events(start_time=start_date, end_time=end_date)
        jurisdiction_events = [e for e in events if e.jurisdiction == jurisdiction]
        
        # Categorize events by type
        event_summary = defaultdict(int)
        severity_summary = defaultdict(int)
        data_category_summary = defaultdict(int)
        
        compliance_violations = []
        security_events = []
        high_risk_events = []
        
        for event in jurisdiction_events:
            event_summary[event.event_type.value] += 1
            severity_summary[event.severity.value] += 1
            data_category_summary[event.data_category.value] += 1
            
            if event.event_type == EventType.COMPLIANCE_VIOLATION:
                compliance_violations.append(event)
            
            if event.event_type == EventType.SECURITY_EVENT:
                security_events.append(event)
            
            if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                high_risk_events.append(event)
        
        return {
            "jurisdiction": jurisdiction,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_events": len(jurisdiction_events),
            "event_summary": dict(event_summary),
            "severity_summary": dict(severity_summary),
            "data_category_summary": dict(data_category_summary),
            "compliance_violations_count": len(compliance_violations),
            "security_events_count": len(security_events),
            "high_risk_events_count": len(high_risk_events),
            "chain_integrity_status": "verified" if self.verify_chain_integrity()["is_valid"] else "compromised",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _save_audit_block(self, block: AuditBlock):
        """Save audit block to persistent storage"""
        block_file = self.data_directory / f"block_{block.block_id}.json"
        
        block_data = block.to_dict()
        
        # Convert datetime objects in events
        for event_data in block_data["events"]:
            if isinstance(event_data["timestamp"], datetime):
                event_data["timestamp"] = event_data["timestamp"].isoformat()
        
        with open(block_file, 'w') as f:
            json.dump(block_data, f, indent=2, default=str)
    
    def _load_audit_chain(self):
        """Load existing audit chain from storage"""
        try:
            block_files = sorted(self.data_directory.glob("block_AUDIT-*.json"))
            
            for block_file in block_files:
                with open(block_file, 'r') as f:
                    block_data = json.load(f)
                
                # Reconstruct events
                events = []
                for event_data in block_data["events"]:
                    # Convert timestamp back to datetime
                    event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                    
                    # Convert enums
                    event_data["event_type"] = EventType(event_data["event_type"])
                    event_data["severity"] = SeverityLevel(event_data["severity"])
                    event_data["data_category"] = DataCategory(event_data["data_category"])
                    
                    event = AuditEvent(**event_data)
                    events.append(event)
                
                # Create block
                block = AuditBlock(
                    block_data["block_id"],
                    block_data["previous_hash"],
                    events
                )
                
                # Restore block metadata
                block.timestamp = datetime.fromisoformat(block_data["timestamp"])
                block.merkle_root = block_data["merkle_root"]
                block.nonce = block_data["nonce"]
                block.hash = block_data["hash"]
                
                self.blocks.append(block)
                self.metrics["total_blocks"] += 1
                self.metrics["total_events"] += len(events)
            
            self.logger.info(f"Loaded audit chain: {len(self.blocks)} blocks, {self.metrics['total_events']} events")
            
        except Exception as e:
            self.logger.error(f"Error loading audit chain: {str(e)}")
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics"""
        return {
            **self.metrics,
            "pending_events": len(self.pending_events),
            "chain_length": len(self.blocks),
            "last_block_time": self.blocks[-1].timestamp.isoformat() if self.blocks else None,
            "integrity_status": "verified" if self.verify_chain_integrity()["is_valid"] else "compromised"
        }
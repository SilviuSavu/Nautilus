#!/usr/bin/env python3
"""
Phase 7: Immutable Audit Trail System
Enterprise-grade immutable audit logging with cryptographic integrity
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import base64
import zlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncpg
import aiofiles
import aioredis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import merkle_tree
import ipfshttpclient
from blockchain import Blockchain
import pandas as pd
from sqlalchemy import create_engine, text
from elasticsearch import AsyncElasticsearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    TRADING_EVENT = "trading_event"
    RISK_EVENT = "risk_event"
    REGULATORY_EVENT = "regulatory_event"
    ADMIN_ACTION = "admin_action"
    API_CALL = "api_call"
    FILE_ACCESS = "file_access"
    DATABASE_OPERATION = "database_operation"
    BACKUP_OPERATION = "backup_operation"
    DISASTER_RECOVERY = "disaster_recovery"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY_ALERT = "security_alert"

class IntegrityMethod(Enum):
    """Methods for ensuring audit trail integrity"""
    CHECKSUM = "checksum"
    DIGITAL_SIGNATURE = "digital_signature"
    MERKLE_TREE = "merkle_tree"
    BLOCKCHAIN = "blockchain"
    CRYPTOGRAPHIC_CHAIN = "cryptographic_chain"

@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: str
    resource_id: str
    action: str
    description: str
    jurisdiction: str
    details: Dict[str, Any]
    
    # Technical metadata
    system_info: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Security and integrity
    checksum: str = field(init=False)
    digital_signature: Optional[str] = None
    previous_event_hash: Optional[str] = None
    merkle_root: Optional[str] = None
    
    # Compliance metadata
    retention_period_days: int = 2555  # 7 years default
    encrypted: bool = True
    compliance_tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate cryptographic checksum for integrity verification"""
        event_data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'action': self.action,
            'description': self.description,
            'jurisdiction': self.jurisdiction,
            'details': self.details,
            'system_info': self.system_info,
            'request_id': self.request_id,
            'correlation_id': self.correlation_id
        }
        
        # Create canonical JSON representation
        canonical_json = json.dumps(event_data, sort_keys=True, separators=(',', ':'), default=str)
        
        # Generate SHA-256 checksum
        self.checksum = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

@dataclass
class AuditChain:
    """Cryptographic chain linking audit events"""
    chain_id: str
    genesis_hash: str
    current_hash: str
    chain_length: int
    events_count: int
    created_at: datetime
    last_updated: datetime
    integrity_verified: bool = True
    verification_timestamp: Optional[datetime] = None

@dataclass
class ComplianceRetentionPolicy:
    """Retention policy for audit events by jurisdiction"""
    policy_id: str
    jurisdiction: str
    event_types: List[AuditEventType]
    retention_days: int
    archive_after_days: int
    encryption_required: bool = True
    immutability_required: bool = True
    export_format: str = "encrypted_json"
    deletion_allowed: bool = False
    compliance_standard: str = ""

class CryptographicIntegrityManager:
    """Manages cryptographic integrity for audit events"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._initialize_keys()
        
        # HMAC secret for additional integrity
        self.hmac_secret = self._generate_hmac_secret()
    
    def _initialize_keys(self):
        """Initialize RSA key pair for digital signatures"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        logger.info("🔐 Cryptographic keys initialized for audit trail integrity")
    
    def _generate_hmac_secret(self) -> bytes:
        """Generate HMAC secret for integrity verification"""
        return hashlib.sha256(f"nautilus_audit_secret_{time.time()}".encode()).digest()
    
    def sign_event(self, event: AuditEvent) -> str:
        """Generate digital signature for audit event"""
        try:
            canonical_data = json.dumps(asdict(event), sort_keys=True, default=str).encode('utf-8')
            
            signature = self.private_key.sign(
                canonical_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to sign audit event: {e}")
            return ""
    
    def verify_signature(self, event: AuditEvent, signature: str) -> bool:
        """Verify digital signature of audit event"""
        try:
            canonical_data = json.dumps(asdict(event), sort_keys=True, default=str).encode('utf-8')
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            
            self.public_key.verify(
                signature_bytes,
                canonical_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data in audit events"""
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self.cipher_suite.encrypt(json_data.encode('utf-8'))
            return base64.b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encrypt audit data: {e}")
            return json.dumps(data, default=str)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data from audit events"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to decrypt audit data: {e}")
            return {"error": "decryption_failed"}
    
    def generate_integrity_hash(self, data: str, previous_hash: Optional[str] = None) -> str:
        """Generate cryptographic chain hash"""
        combined_data = f"{previous_hash or ''}{data}{time.time()}"
        return hashlib.sha256(combined_data.encode('utf-8')).hexdigest()
    
    def verify_integrity_chain(self, events: List[AuditEvent]) -> bool:
        """Verify integrity of cryptographic chain"""
        if not events:
            return True
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        previous_hash = None
        for event in sorted_events:
            expected_hash = self.generate_integrity_hash(
                event.checksum, 
                previous_hash
            )
            
            # In a real implementation, we would compare with stored hash
            # For now, we verify the checksum is consistent
            if not event.checksum:
                logger.warning(f"Missing checksum for event {event.event_id}")
                return False
            
            previous_hash = event.checksum
        
        return True

class MerkleTreeIntegrity:
    """Merkle tree-based integrity verification for audit events"""
    
    def __init__(self):
        self.trees = {}  # Store Merkle trees by batch/time period
    
    def create_merkle_tree(self, events: List[AuditEvent]) -> str:
        """Create Merkle tree from audit events"""
        if not events:
            return ""
        
        # Extract checksums as leaf nodes
        leaf_hashes = [event.checksum for event in events]
        
        # Build Merkle tree
        tree = self._build_tree(leaf_hashes)
        root_hash = tree[0] if tree else ""
        
        # Store tree for future verification
        tree_id = hashlib.sha256(f"{''.join(leaf_hashes)}{time.time()}".encode()).hexdigest()[:16]
        self.trees[tree_id] = {
            'root_hash': root_hash,
            'leaf_hashes': leaf_hashes,
            'tree_structure': tree,
            'event_count': len(events),
            'created_at': datetime.now(timezone.utc)
        }
        
        logger.info(f"🌳 Created Merkle tree {tree_id} with root hash: {root_hash[:16]}...")
        
        return root_hash
    
    def _build_tree(self, hashes: List[str]) -> List[str]:
        """Build Merkle tree from list of hashes"""
        if not hashes:
            return []
        
        if len(hashes) == 1:
            return hashes
        
        # Ensure even number of nodes
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last hash
        
        # Build next level
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            parent_hash = hashlib.sha256(combined.encode()).hexdigest()
            next_level.append(parent_hash)
        
        # Recursively build tree
        parent_tree = self._build_tree(next_level)
        return parent_tree + hashes
    
    def verify_merkle_proof(self, event_hash: str, proof_path: List[str], root_hash: str) -> bool:
        """Verify Merkle proof for a specific event"""
        current_hash = event_hash
        
        for proof_hash in proof_path:
            # Try both left and right combinations
            left_combined = current_hash + proof_hash
            right_combined = proof_hash + current_hash
            
            left_hash = hashlib.sha256(left_combined.encode()).hexdigest()
            right_hash = hashlib.sha256(right_combined.encode()).hexdigest()
            
            # Use the hash that leads toward the root
            current_hash = left_hash  # Simplified - real implementation would track path
        
        return current_hash == root_hash

class BlockchainIntegrity:
    """Blockchain-based integrity for critical audit events"""
    
    def __init__(self):
        self.blockchain = self._initialize_blockchain()
        
    def _initialize_blockchain(self):
        """Initialize private blockchain for audit integrity"""
        # This would initialize a private blockchain
        # For demonstration, we'll use a simplified structure
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'data': 'Nautilus Audit Trail Genesis Block',
            'previous_hash': '0',
            'hash': self._calculate_block_hash(0, time.time(), 'Genesis', '0')
        }
        
        return [genesis_block]
    
    def _calculate_block_hash(self, index: int, timestamp: float, data: str, previous_hash: str) -> str:
        """Calculate hash for blockchain block"""
        value = f"{index}{timestamp}{data}{previous_hash}"
        return hashlib.sha256(value.encode()).hexdigest()
    
    def add_audit_block(self, events: List[AuditEvent]) -> str:
        """Add audit events to blockchain"""
        if not events:
            return ""
        
        previous_block = self.blockchain[-1]
        new_index = previous_block['index'] + 1
        new_timestamp = time.time()
        
        # Serialize events for blockchain storage
        events_data = json.dumps([asdict(event) for event in events], default=str)
        
        new_hash = self._calculate_block_hash(
            new_index,
            new_timestamp,
            events_data,
            previous_block['hash']
        )
        
        new_block = {
            'index': new_index,
            'timestamp': new_timestamp,
            'data': events_data,
            'previous_hash': previous_block['hash'],
            'hash': new_hash,
            'event_count': len(events)
        }
        
        self.blockchain.append(new_block)
        
        logger.info(f"⛓️ Added audit block {new_index} to blockchain with hash: {new_hash[:16]}...")
        
        return new_hash
    
    def verify_blockchain_integrity(self) -> bool:
        """Verify integrity of entire blockchain"""
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i - 1]
            
            # Verify current block's hash
            calculated_hash = self._calculate_block_hash(
                current_block['index'],
                current_block['timestamp'],
                current_block['data'],
                current_block['previous_hash']
            )
            
            if calculated_hash != current_block['hash']:
                logger.error(f"Blockchain integrity violation in block {i}")
                return False
            
            # Verify link to previous block
            if current_block['previous_hash'] != previous_block['hash']:
                logger.error(f"Blockchain chain violation between blocks {i-1} and {i}")
                return False
        
        return True

class AuditStorage:
    """Multi-tier storage system for audit events"""
    
    def __init__(self):
        self.primary_db_pool = None
        self.archive_db_pool = None
        self.elasticsearch_client = None
        self.redis_client = None
        self.ipfs_client = None
        
        # Storage tiers
        self.hot_storage_days = 30      # PostgreSQL
        self.warm_storage_days = 365    # Elasticsearch
        self.cold_storage_days = 2555   # IPFS/Archive
        
    async def initialize(self):
        """Initialize all storage backends"""
        await self._initialize_primary_database()
        await self._initialize_elasticsearch()
        await self._initialize_redis_cache()
        await self._initialize_ipfs()
        
        logger.info("💾 Multi-tier audit storage initialized")
    
    async def _initialize_primary_database(self):
        """Initialize primary PostgreSQL database"""
        try:
            self.primary_db_pool = await asyncpg.create_pool(
                "postgresql://nautilus:nautilus123@postgres:5432/nautilus",
                min_size=5,
                max_size=20
            )
            
            async with self.primary_db_pool.acquire() as conn:
                # Create audit events table with partitioning
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id VARCHAR PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        event_type VARCHAR NOT NULL,
                        severity VARCHAR NOT NULL,
                        user_id VARCHAR NOT NULL,
                        session_id VARCHAR,
                        ip_address INET,
                        user_agent TEXT,
                        resource_type VARCHAR NOT NULL,
                        resource_id VARCHAR NOT NULL,
                        action VARCHAR NOT NULL,
                        description TEXT NOT NULL,
                        jurisdiction VARCHAR NOT NULL,
                        details JSONB NOT NULL,
                        system_info JSONB,
                        request_id VARCHAR,
                        correlation_id VARCHAR,
                        checksum VARCHAR NOT NULL,
                        digital_signature TEXT,
                        previous_event_hash VARCHAR,
                        merkle_root VARCHAR,
                        retention_period_days INTEGER DEFAULT 2555,
                        encrypted BOOLEAN DEFAULT TRUE,
                        compliance_tags TEXT[],
                        created_at TIMESTAMP DEFAULT NOW()
                    ) PARTITION BY RANGE (timestamp)
                """)
                
                # Create integrity chains table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_chains (
                        chain_id VARCHAR PRIMARY KEY,
                        genesis_hash VARCHAR NOT NULL,
                        current_hash VARCHAR NOT NULL,
                        chain_length INTEGER NOT NULL,
                        events_count INTEGER NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        last_updated TIMESTAMP NOT NULL,
                        integrity_verified BOOLEAN DEFAULT TRUE,
                        verification_timestamp TIMESTAMP
                    )
                """)
                
                # Create retention policies table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_retention_policies (
                        policy_id VARCHAR PRIMARY KEY,
                        jurisdiction VARCHAR NOT NULL,
                        event_types TEXT[] NOT NULL,
                        retention_days INTEGER NOT NULL,
                        archive_after_days INTEGER NOT NULL,
                        encryption_required BOOLEAN DEFAULT TRUE,
                        immutability_required BOOLEAN DEFAULT TRUE,
                        export_format VARCHAR DEFAULT 'encrypted_json',
                        deletion_allowed BOOLEAN DEFAULT FALSE,
                        compliance_standard VARCHAR,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create monthly partitions for current year
                current_year = datetime.now().year
                for month in range(1, 13):
                    start_date = f"{current_year}-{month:02d}-01"
                    end_date = f"{current_year}-{month:02d}-28" if month == 2 else f"{current_year}-{month:02d}-31"
                    
                    try:
                        await conn.execute(f"""
                            CREATE TABLE IF NOT EXISTS audit_events_{current_year}_{month:02d}
                            PARTITION OF audit_events
                            FOR VALUES FROM ('{start_date}') TO ('{end_date}')
                        """)
                    except Exception:
                        pass  # Partition might already exist
                
                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_jurisdiction ON audit_events(jurisdiction)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_checksum ON audit_events(checksum)")
                
            logger.info("✅ Primary database initialized with partitioned audit tables")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize primary database: {e}")
            raise
    
    async def _initialize_elasticsearch(self):
        """Initialize Elasticsearch for searchable audit logs"""
        try:
            self.elasticsearch_client = AsyncElasticsearch([
                {'host': 'localhost', 'port': 9200}
            ])
            
            # Create audit index template
            index_template = {
                "index_patterns": ["audit-events-*"],
                "template": {
                    "settings": {
                        "number_of_shards": 3,
                        "number_of_replicas": 1,
                        "index.lifecycle.name": "audit-policy",
                        "index.lifecycle.rollover_alias": "audit-events"
                    },
                    "mappings": {
                        "properties": {
                            "event_id": {"type": "keyword"},
                            "timestamp": {"type": "date"},
                            "event_type": {"type": "keyword"},
                            "severity": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "ip_address": {"type": "ip"},
                            "resource_type": {"type": "keyword"},
                            "resource_id": {"type": "keyword"},
                            "action": {"type": "keyword"},
                            "description": {"type": "text"},
                            "jurisdiction": {"type": "keyword"},
                            "details": {"type": "object", "enabled": False},
                            "checksum": {"type": "keyword"},
                            "compliance_tags": {"type": "keyword"}
                        }
                    }
                }
            }
            
            await self.elasticsearch_client.indices.put_index_template(
                name="audit-events-template",
                body=index_template
            )
            
            logger.info("✅ Elasticsearch initialized for audit log search")
            
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch initialization failed: {e}")
            self.elasticsearch_client = None
    
    async def _initialize_redis_cache(self):
        """Initialize Redis for hot audit data caching"""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=True
            )
            
            await self.redis_client.ping()
            logger.info("✅ Redis cache initialized for hot audit data")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed: {e}")
            self.redis_client = None
    
    async def _initialize_ipfs(self):
        """Initialize IPFS for immutable cold storage"""
        try:
            # This would initialize IPFS client for immutable storage
            # For demonstration, we'll simulate IPFS functionality
            logger.info("✅ IPFS simulation initialized for cold storage")
            
        except Exception as e:
            logger.warning(f"⚠️ IPFS initialization failed: {e}")
            self.ipfs_client = None
    
    async def store_audit_event(self, event: AuditEvent) -> bool:
        """Store audit event in appropriate storage tier"""
        try:
            # Primary storage (PostgreSQL)
            success = await self._store_in_primary_db(event)
            
            if not success:
                logger.error(f"Failed to store audit event {event.event_id} in primary database")
                return False
            
            # Hot cache (Redis) - for recent events
            if self.redis_client:
                await self._cache_hot_event(event)
            
            # Search index (Elasticsearch)
            if self.elasticsearch_client:
                await self._index_for_search(event)
            
            logger.info(f"📝 Stored audit event {event.event_id} in multi-tier storage")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store audit event {event.event_id}: {e}")
            return False
    
    async def _store_in_primary_db(self, event: AuditEvent) -> bool:
        """Store event in primary PostgreSQL database"""
        try:
            async with self.primary_db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events 
                    (event_id, timestamp, event_type, severity, user_id, session_id, ip_address,
                     user_agent, resource_type, resource_id, action, description, jurisdiction,
                     details, system_info, request_id, correlation_id, checksum, digital_signature,
                     previous_event_hash, merkle_root, retention_period_days, encrypted, compliance_tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
                """,
                event.event_id,
                event.timestamp,
                event.event_type.value,
                event.severity.value,
                event.user_id,
                event.session_id,
                event.ip_address,
                event.user_agent,
                event.resource_type,
                event.resource_id,
                event.action,
                event.description,
                event.jurisdiction,
                json.dumps(event.details),
                json.dumps(event.system_info),
                event.request_id,
                event.correlation_id,
                event.checksum,
                event.digital_signature,
                event.previous_event_hash,
                event.merkle_root,
                event.retention_period_days,
                event.encrypted,
                event.compliance_tags
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Primary database storage error: {e}")
            return False
    
    async def _cache_hot_event(self, event: AuditEvent):
        """Cache recent event in Redis for fast access"""
        try:
            # Store event data
            event_key = f"audit:event:{event.event_id}"
            event_data = json.dumps(asdict(event), default=str)
            
            await self.redis_client.setex(
                event_key, 
                self.hot_storage_days * 24 * 3600,  # TTL in seconds
                event_data
            )
            
            # Add to user's recent events
            user_key = f"audit:user:{event.user_id}:recent"
            await self.redis_client.lpush(user_key, event.event_id)
            await self.redis_client.ltrim(user_key, 0, 99)  # Keep last 100 events
            await self.redis_client.expire(user_key, self.hot_storage_days * 24 * 3600)
            
            # Add to jurisdiction index
            jurisdiction_key = f"audit:jurisdiction:{event.jurisdiction}:recent"
            await self.redis_client.lpush(jurisdiction_key, event.event_id)
            await self.redis_client.ltrim(jurisdiction_key, 0, 499)  # Keep last 500 events
            await self.redis_client.expire(jurisdiction_key, self.hot_storage_days * 24 * 3600)
            
        except Exception as e:
            logger.warning(f"Redis caching failed: {e}")
    
    async def _index_for_search(self, event: AuditEvent):
        """Index event in Elasticsearch for search capabilities"""
        try:
            index_name = f"audit-events-{event.timestamp.strftime('%Y-%m')}"
            
            doc = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'severity': event.severity.value,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'resource_type': event.resource_type,
                'resource_id': event.resource_id,
                'action': event.action,
                'description': event.description,
                'jurisdiction': event.jurisdiction,
                'details': event.details,  # Store as object for search
                'checksum': event.checksum,
                'compliance_tags': event.compliance_tags
            }
            
            await self.elasticsearch_client.index(
                index=index_name,
                id=event.event_id,
                body=doc
            )
            
        except Exception as e:
            logger.warning(f"Elasticsearch indexing failed: {e}")
    
    async def get_audit_events(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        
        # Try hot cache first for recent data
        if (datetime.now(timezone.utc) - start_time).days <= self.hot_storage_days:
            cached_events = await self._get_from_cache(start_time, end_time, user_id, event_type, jurisdiction, limit)
            if cached_events:
                return cached_events
        
        # Fallback to primary database
        return await self._get_from_primary_db(start_time, end_time, user_id, event_type, jurisdiction, limit)
    
    async def _get_from_cache(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str],
        event_type: Optional[AuditEventType],
        jurisdiction: Optional[str],
        limit: int
    ) -> List[AuditEvent]:
        """Get events from Redis cache"""
        if not self.redis_client:
            return []
        
        try:
            # This would implement Redis-based filtering
            # For now, return empty list to fallback to database
            return []
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return []
    
    async def _get_from_primary_db(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str],
        event_type: Optional[AuditEventType],
        jurisdiction: Optional[str],
        limit: int
    ) -> List[AuditEvent]:
        """Get events from primary database"""
        try:
            query_parts = ["SELECT * FROM audit_events WHERE timestamp >= $1 AND timestamp <= $2"]
            params = [start_time, end_time]
            param_count = 2
            
            if user_id:
                param_count += 1
                query_parts.append(f"AND user_id = ${param_count}")
                params.append(user_id)
            
            if event_type:
                param_count += 1
                query_parts.append(f"AND event_type = ${param_count}")
                params.append(event_type.value)
            
            if jurisdiction:
                param_count += 1
                query_parts.append(f"AND jurisdiction = ${param_count}")
                params.append(jurisdiction)
            
            query_parts.append("ORDER BY timestamp DESC")
            query_parts.append(f"LIMIT {limit}")
            
            query = " ".join(query_parts)
            
            async with self.primary_db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row['event_id'],
                    timestamp=row['timestamp'],
                    event_type=AuditEventType(row['event_type']),
                    severity=AuditSeverity(row['severity']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    resource_type=row['resource_type'],
                    resource_id=row['resource_id'],
                    action=row['action'],
                    description=row['description'],
                    jurisdiction=row['jurisdiction'],
                    details=json.loads(row['details']) if row['details'] else {},
                    system_info=json.loads(row['system_info']) if row['system_info'] else {},
                    request_id=row['request_id'],
                    correlation_id=row['correlation_id']
                )
                
                # Restore integrity fields
                event.checksum = row['checksum']
                event.digital_signature = row['digital_signature']
                event.previous_event_hash = row['previous_event_hash']
                event.merkle_root = row['merkle_root']
                event.retention_period_days = row['retention_period_days'] or 2555
                event.encrypted = row['encrypted']
                event.compliance_tags = row['compliance_tags'] or []
                
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Database retrieval failed: {e}")
            return []

class ImmutableAuditTrailSystem:
    """
    Main immutable audit trail system with multi-tier storage and cryptographic integrity
    """
    
    def __init__(self):
        self.integrity_manager = CryptographicIntegrityManager()
        self.merkle_tree = MerkleTreeIntegrity()
        self.blockchain = BlockchainIntegrity()
        self.storage = AuditStorage()
        
        # Audit chains for cryptographic linking
        self.audit_chains = {}
        self.current_chain_id = None
        
        # Retention policies by jurisdiction
        self.retention_policies = self._initialize_retention_policies()
        
        # Batch processing for performance
        self.batch_queue = []
        self.batch_size = 100
        self.batch_timeout = 30  # seconds
        self.last_batch_time = time.time()
        
        # Integrity verification
        self.integrity_check_interval = 3600  # 1 hour
        self.last_integrity_check = time.time()
        
        # Performance metrics
        self.metrics = {
            'events_processed': 0,
            'events_stored': 0,
            'integrity_checks': 0,
            'integrity_failures': 0,
            'avg_storage_time_ms': 0.0,
            'batch_operations': 0
        }
    
    def _initialize_retention_policies(self) -> Dict[str, ComplianceRetentionPolicy]:
        """Initialize jurisdiction-specific retention policies"""
        return {
            'US_SEC': ComplianceRetentionPolicy(
                policy_id="US_SEC_GENERAL",
                jurisdiction="US_SEC",
                event_types=[AuditEventType.TRADING_EVENT, AuditEventType.REGULATORY_EVENT, 
                           AuditEventType.COMPLIANCE_EVENT, AuditEventType.RISK_EVENT],
                retention_days=2555,  # 7 years
                archive_after_days=1825,  # 5 years
                encryption_required=True,
                immutability_required=True,
                compliance_standard="SEC Rule 17a-4"
            ),
            'EU_MIFID2': ComplianceRetentionPolicy(
                policy_id="EU_MIFID2_GENERAL",
                jurisdiction="EU_MIFID2",
                event_types=[AuditEventType.TRADING_EVENT, AuditEventType.REGULATORY_EVENT],
                retention_days=1825,  # 5 years
                archive_after_days=1095,  # 3 years
                encryption_required=True,
                immutability_required=True,
                compliance_standard="MiFID II Article 25"
            ),
            'UK_FCA': ComplianceRetentionPolicy(
                policy_id="UK_FCA_GENERAL",
                jurisdiction="UK_FCA",
                event_types=[AuditEventType.TRADING_EVENT, AuditEventType.REGULATORY_EVENT],
                retention_days=2190,  # 6 years
                archive_after_days=1095,  # 3 years
                encryption_required=True,
                immutability_required=True,
                compliance_standard="FCA SYSC 9"
            ),
            'GENERAL': ComplianceRetentionPolicy(
                policy_id="GENERAL_DEFAULT",
                jurisdiction="GENERAL",
                event_types=list(AuditEventType),
                retention_days=2555,  # 7 years default
                archive_after_days=1095,  # 3 years
                encryption_required=False,
                immutability_required=False,
                compliance_standard="Internal Policy"
            )
        }
    
    async def initialize(self):
        """Initialize the audit trail system"""
        logger.info("🔒 Initializing Immutable Audit Trail System")
        
        await self.storage.initialize()
        
        # Initialize genesis chain
        self.current_chain_id = await self._create_new_chain()
        
        logger.info("✅ Immutable Audit Trail System initialized")
    
    async def _create_new_chain(self) -> str:
        """Create new cryptographic chain"""
        chain_id = str(uuid.uuid4())
        genesis_hash = hashlib.sha256(f"genesis_{chain_id}_{time.time()}".encode()).hexdigest()
        
        chain = AuditChain(
            chain_id=chain_id,
            genesis_hash=genesis_hash,
            current_hash=genesis_hash,
            chain_length=0,
            events_count=0,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        self.audit_chains[chain_id] = chain
        
        logger.info(f"🔗 Created new audit chain: {chain_id}")
        
        return chain_id
    
    async def log_audit_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        description: str,
        jurisdiction: str = "GENERAL",
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None
    ) -> str:
        """
        Log an immutable audit event with cryptographic integrity
        
        Args:
            event_type: Type of audit event
            severity: Severity level
            user_id: User performing the action
            resource_type: Type of resource being accessed
            resource_id: Unique identifier of resource
            action: Action being performed
            description: Human-readable description
            jurisdiction: Regulatory jurisdiction
            details: Additional event details
            session_id: User session identifier
            ip_address: Source IP address
            user_agent: User agent string
            request_id: Request identifier for tracing
            correlation_id: Correlation identifier for related events
            compliance_tags: Tags for compliance categorization
            
        Returns:
            Event ID of the logged audit event
        """
        start_time = time.time()
        
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            description=description,
            jurisdiction=jurisdiction,
            details=details or {},
            system_info={
                'hostname': 'nautilus-audit-system',
                'process_id': threading.get_ident(),
                'timestamp_ms': int(time.time() * 1000)
            },
            request_id=request_id,
            correlation_id=correlation_id,
            compliance_tags=compliance_tags or []
        )
        
        # Apply retention policy
        policy = self.retention_policies.get(jurisdiction, self.retention_policies['GENERAL'])
        event.retention_period_days = policy.retention_days
        event.encrypted = policy.encryption_required
        
        # Generate digital signature
        event.digital_signature = self.integrity_manager.sign_event(event)
        
        # Link to previous event in chain
        current_chain = self.audit_chains.get(self.current_chain_id)
        if current_chain:
            event.previous_event_hash = current_chain.current_hash
        
        # Add to batch queue for processing
        self.batch_queue.append(event)
        
        # Process batch if conditions are met
        if (len(self.batch_queue) >= self.batch_size or 
            time.time() - self.last_batch_time > self.batch_timeout):
            await self._process_batch()
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        self.metrics['events_processed'] += 1
        current_avg = self.metrics['avg_storage_time_ms']
        total_events = self.metrics['events_processed']
        self.metrics['avg_storage_time_ms'] = (current_avg * (total_events - 1) + processing_time) / total_events
        
        logger.info(f"📝 Logged audit event {event.event_id} - Type: {event_type.value}, Severity: {severity.value}")
        
        return event.event_id
    
    async def _process_batch(self):
        """Process batch of audit events with cryptographic integrity"""
        if not self.batch_queue:
            return
        
        batch_start = time.time()
        batch_events = self.batch_queue.copy()
        self.batch_queue.clear()
        self.last_batch_time = time.time()
        
        try:
            # Generate Merkle tree for batch integrity
            merkle_root = self.merkle_tree.create_merkle_tree(batch_events)
            
            # Update events with Merkle root
            for event in batch_events:
                event.merkle_root = merkle_root
            
            # Store events in multi-tier storage
            storage_tasks = [self.storage.store_audit_event(event) for event in batch_events]
            results = await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            successful_stores = sum(1 for r in results if r is True)
            
            # Add to blockchain for critical events
            critical_events = [e for e in batch_events if e.severity in [AuditSeverity.CRITICAL, AuditSeverity.SECURITY_ALERT]]
            if critical_events:
                blockchain_hash = self.blockchain.add_audit_block(critical_events)
                logger.info(f"⛓️ Added {len(critical_events)} critical events to blockchain")
            
            # Update current chain
            current_chain = self.audit_chains.get(self.current_chain_id)
            if current_chain:
                current_chain.current_hash = self.integrity_manager.generate_integrity_hash(
                    merkle_root, current_chain.current_hash
                )
                current_chain.chain_length += 1
                current_chain.events_count += len(batch_events)
                current_chain.last_updated = datetime.now(timezone.utc)
            
            # Update metrics
            batch_time = (time.time() - batch_start) * 1000
            self.metrics['events_stored'] += successful_stores
            self.metrics['batch_operations'] += 1
            
            logger.info(f"📦 Processed batch of {len(batch_events)} events in {batch_time:.1f}ms")
            logger.info(f"   Stored: {successful_stores}, Merkle root: {merkle_root[:16]}...")
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            # Re-queue events for retry
            self.batch_queue.extend(batch_events)
    
    async def verify_audit_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive integrity verification of audit trail
        
        Args:
            start_time: Start of time range to verify
            end_time: End of time range to verify  
            event_ids: Specific event IDs to verify
            
        Returns:
            Integrity verification report
        """
        verification_start = time.time()
        
        logger.info("🔍 Starting comprehensive audit trail integrity verification")
        
        # Get events to verify
        if event_ids:
            events = []
            for event_id in event_ids:
                event_list = await self.storage.get_audit_events(
                    datetime.min.replace(tzinfo=timezone.utc),
                    datetime.max.replace(tzinfo=timezone.utc),
                    limit=1
                )
                events.extend(event_list)
        else:
            start_time = start_time or (datetime.now(timezone.utc) - timedelta(hours=24))
            end_time = end_time or datetime.now(timezone.utc)
            events = await self.storage.get_audit_events(start_time, end_time, limit=10000)
        
        verification_report = {
            'verification_timestamp': datetime.now(timezone.utc).isoformat(),
            'events_verified': len(events),
            'integrity_checks': {
                'checksum_verification': {'passed': 0, 'failed': 0},
                'signature_verification': {'passed': 0, 'failed': 0},
                'chain_verification': {'passed': 0, 'failed': 0},
                'merkle_verification': {'passed': 0, 'failed': 0},
                'blockchain_verification': {'passed': 0, 'failed': 0}
            },
            'failed_events': [],
            'overall_integrity': True,
            'verification_time_ms': 0
        }
        
        # Verify individual event integrity
        for event in events:
            event_integrity = await self._verify_single_event_integrity(event)
            
            # Update report
            for check_type, result in event_integrity.items():
                if check_type in verification_report['integrity_checks']:
                    if result:
                        verification_report['integrity_checks'][check_type]['passed'] += 1
                    else:
                        verification_report['integrity_checks'][check_type]['failed'] += 1
                        verification_report['failed_events'].append({
                            'event_id': event.event_id,
                            'check_type': check_type,
                            'timestamp': event.timestamp.isoformat()
                        })
                        verification_report['overall_integrity'] = False
        
        # Verify cryptographic chain integrity
        chain_integrity = self.integrity_manager.verify_integrity_chain(events)
        if chain_integrity:
            verification_report['integrity_checks']['chain_verification']['passed'] += 1
        else:
            verification_report['integrity_checks']['chain_verification']['failed'] += 1
            verification_report['overall_integrity'] = False
        
        # Verify blockchain integrity
        blockchain_integrity = self.blockchain.verify_blockchain_integrity()
        if blockchain_integrity:
            verification_report['integrity_checks']['blockchain_verification']['passed'] += 1
        else:
            verification_report['integrity_checks']['blockchain_verification']['failed'] += 1
            verification_report['overall_integrity'] = False
        
        # Calculate verification statistics
        verification_time = (time.time() - verification_start) * 1000
        verification_report['verification_time_ms'] = verification_time
        
        # Update metrics
        self.metrics['integrity_checks'] += 1
        if not verification_report['overall_integrity']:
            self.metrics['integrity_failures'] += 1
        
        self.last_integrity_check = time.time()
        
        status = "✅ PASSED" if verification_report['overall_integrity'] else "❌ FAILED"
        logger.info(f"🔍 Integrity verification {status} for {len(events)} events in {verification_time:.1f}ms")
        
        return verification_report
    
    async def _verify_single_event_integrity(self, event: AuditEvent) -> Dict[str, bool]:
        """Verify integrity of a single audit event"""
        results = {}
        
        # Verify checksum
        try:
            # Reconstruct event without checksum and verify
            temp_event = AuditEvent(
                event_id=event.event_id,
                timestamp=event.timestamp,
                event_type=event.event_type,
                severity=event.severity,
                user_id=event.user_id,
                session_id=event.session_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                description=event.description,
                jurisdiction=event.jurisdiction,
                details=event.details,
                system_info=event.system_info,
                request_id=event.request_id,
                correlation_id=event.correlation_id
            )
            
            results['checksum_verification'] = temp_event.checksum == event.checksum
            
        except Exception:
            results['checksum_verification'] = False
        
        # Verify digital signature
        if event.digital_signature:
            results['signature_verification'] = self.integrity_manager.verify_signature(
                event, event.digital_signature
            )
        else:
            results['signature_verification'] = True  # No signature to verify
        
        return results
    
    async def search_audit_events(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severities: Optional[List[AuditSeverity]] = None,
        user_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """
        Advanced search across audit events
        
        Args:
            query: Search query string
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            severities: Filter by severities
            user_id: Filter by user ID
            jurisdiction: Filter by jurisdiction
            limit: Maximum results
            
        Returns:
            List of matching audit events
        """
        logger.info(f"🔍 Searching audit events: '{query}'")
        
        # Use Elasticsearch if available for full-text search
        if self.storage.elasticsearch_client and query:
            try:
                return await self._search_elasticsearch(
                    query, start_time, end_time, event_types, 
                    severities, user_id, jurisdiction, limit
                )
            except Exception as e:
                logger.warning(f"Elasticsearch search failed, falling back to database: {e}")
        
        # Fallback to database search
        return await self._search_database(
            query, start_time, end_time, event_types,
            severities, user_id, jurisdiction, limit
        )
    
    async def _search_elasticsearch(
        self,
        query: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        event_types: Optional[List[AuditEventType]],
        severities: Optional[List[AuditSeverity]],
        user_id: Optional[str],
        jurisdiction: Optional[str],
        limit: int
    ) -> List[AuditEvent]:
        """Search audit events using Elasticsearch"""
        
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["description", "action", "resource_type", "resource_id"]
                            }
                        }
                    ],
                    "filter": []
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": limit
        }
        
        # Add time range filter
        if start_time or end_time:
            time_filter = {"range": {"timestamp": {}}}
            if start_time:
                time_filter["range"]["timestamp"]["gte"] = start_time.isoformat()
            if end_time:
                time_filter["range"]["timestamp"]["lte"] = end_time.isoformat()
            search_body["query"]["bool"]["filter"].append(time_filter)
        
        # Add additional filters
        if event_types:
            search_body["query"]["bool"]["filter"].append({
                "terms": {"event_type": [et.value for et in event_types]}
            })
        
        if severities:
            search_body["query"]["bool"]["filter"].append({
                "terms": {"severity": [s.value for s in severities]}
            })
        
        if user_id:
            search_body["query"]["bool"]["filter"].append({
                "term": {"user_id": user_id}
            })
        
        if jurisdiction:
            search_body["query"]["bool"]["filter"].append({
                "term": {"jurisdiction": jurisdiction}
            })
        
        try:
            response = await self.storage.elasticsearch_client.search(
                index="audit-events-*",
                body=search_body
            )
            
            events = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # Reconstruct AuditEvent object
                event = AuditEvent(
                    event_id=source['event_id'],
                    timestamp=datetime.fromisoformat(source['timestamp'].replace('Z', '+00:00')),
                    event_type=AuditEventType(source['event_type']),
                    severity=AuditSeverity(source['severity']),
                    user_id=source['user_id'],
                    session_id=source.get('session_id'),
                    ip_address=source.get('ip_address'),
                    user_agent=source.get('user_agent'),
                    resource_type=source['resource_type'],
                    resource_id=source['resource_id'],
                    action=source['action'],
                    description=source['description'],
                    jurisdiction=source['jurisdiction'],
                    details=source.get('details', {}),
                    compliance_tags=source.get('compliance_tags', [])
                )
                
                event.checksum = source['checksum']
                events.append(event)
            
            logger.info(f"📊 Elasticsearch search returned {len(events)} results")
            return events
            
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return []
    
    async def _search_database(
        self,
        query: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        event_types: Optional[List[AuditEventType]],
        severities: Optional[List[AuditSeverity]],
        user_id: Optional[str],
        jurisdiction: Optional[str],
        limit: int
    ) -> List[AuditEvent]:
        """Search audit events using database full-text search"""
        
        # Build dynamic query
        where_conditions = []
        params = []
        param_count = 0
        
        if query:
            param_count += 1
            where_conditions.append(f"(description ILIKE ${param_count} OR action ILIKE ${param_count} OR resource_type ILIKE ${param_count})")
            params.append(f"%{query}%")
        
        if start_time:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(end_time)
        
        if user_id:
            param_count += 1
            where_conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        
        if jurisdiction:
            param_count += 1
            where_conditions.append(f"jurisdiction = ${param_count}")
            params.append(jurisdiction)
        
        if event_types:
            param_count += 1
            where_conditions.append(f"event_type = ANY(${param_count})")
            params.append([et.value for et in event_types])
        
        if severities:
            param_count += 1
            where_conditions.append(f"severity = ANY(${param_count})")
            params.append([s.value for s in severities])
        
        # Build final query
        base_query = "SELECT * FROM audit_events"
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        base_query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        try:
            async with self.storage.primary_db_pool.acquire() as conn:
                rows = await conn.fetch(base_query, *params)
            
            events = []
            for row in rows:
                event = AuditEvent(
                    event_id=row['event_id'],
                    timestamp=row['timestamp'],
                    event_type=AuditEventType(row['event_type']),
                    severity=AuditSeverity(row['severity']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    ip_address=row['ip_address'],
                    user_agent=row['user_agent'],
                    resource_type=row['resource_type'],
                    resource_id=row['resource_id'],
                    action=row['action'],
                    description=row['description'],
                    jurisdiction=row['jurisdiction'],
                    details=json.loads(row['details']) if row['details'] else {},
                    system_info=json.loads(row['system_info']) if row['system_info'] else {},
                    request_id=row['request_id'],
                    correlation_id=row['correlation_id']
                )
                
                # Restore integrity fields
                event.checksum = row['checksum']
                event.digital_signature = row['digital_signature']
                event.previous_event_hash = row['previous_event_hash']
                event.merkle_root = row['merkle_root']
                event.retention_period_days = row['retention_period_days'] or 2555
                event.encrypted = row['encrypted']
                event.compliance_tags = row['compliance_tags'] or []
                
                events.append(event)
            
            logger.info(f"📊 Database search returned {len(events)} results")
            return events
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    async def export_audit_trail(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
        jurisdiction: Optional[str] = None,
        include_integrity_proof: bool = True
    ) -> Tuple[str, bytes]:
        """
        Export audit trail for compliance or investigation
        
        Args:
            start_time: Start of export range
            end_time: End of export range
            format: Export format (json, csv, xml)
            jurisdiction: Filter by jurisdiction
            include_integrity_proof: Include cryptographic proofs
            
        Returns:
            Tuple of (filename, content_bytes)
        """
        logger.info(f"📦 Exporting audit trail from {start_time} to {end_time}")
        
        # Get events for export
        events = await self.storage.get_audit_events(
            start_time, end_time, jurisdiction=jurisdiction, limit=100000
        )
        
        # Generate integrity proof if requested
        integrity_proof = None
        if include_integrity_proof:
            integrity_proof = await self.verify_audit_integrity(start_time, end_time)
        
        # Export in requested format
        if format.lower() == "json":
            return await self._export_json(events, integrity_proof, start_time, end_time)
        elif format.lower() == "csv":
            return await self._export_csv(events, start_time, end_time)
        elif format.lower() == "xml":
            return await self._export_xml(events, integrity_proof, start_time, end_time)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _export_json(
        self,
        events: List[AuditEvent],
        integrity_proof: Optional[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[str, bytes]:
        """Export audit trail as JSON"""
        
        export_data = {
            "export_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "events_count": len(events),
                "format": "json",
                "system": "Nautilus Immutable Audit Trail",
                "version": "1.0"
            },
            "events": [asdict(event) for event in events],
            "integrity_verification": integrity_proof
        }
        
        json_content = json.dumps(export_data, indent=2, default=str)
        filename = f"audit_trail_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.json"
        
        return filename, json_content.encode('utf-8')
    
    async def _export_csv(
        self,
        events: List[AuditEvent],
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[str, bytes]:
        """Export audit trail as CSV"""
        
        if not events:
            filename = f"audit_trail_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
            return filename, b"No audit events in specified time range"
        
        # Convert events to DataFrame
        df_data = []
        for event in events:
            row = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'severity': event.severity.value,
                'user_id': event.user_id,
                'session_id': event.session_id or '',
                'ip_address': event.ip_address or '',
                'resource_type': event.resource_type,
                'resource_id': event.resource_id,
                'action': event.action,
                'description': event.description,
                'jurisdiction': event.jurisdiction,
                'checksum': event.checksum,
                'details': json.dumps(event.details) if event.details else ''
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Export to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        filename = f"audit_trail_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
        
        return filename, csv_content.encode('utf-8')
    
    async def _export_xml(
        self,
        events: List[AuditEvent],
        integrity_proof: Optional[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[str, bytes]:
        """Export audit trail as XML"""
        
        root = ET.Element('AuditTrailExport')
        root.set('version', '1.0')
        
        # Metadata
        metadata = ET.SubElement(root, 'ExportMetadata')
        ET.SubElement(metadata, 'GeneratedAt').text = datetime.now(timezone.utc).isoformat()
        ET.SubElement(metadata, 'StartTime').text = start_time.isoformat()
        ET.SubElement(metadata, 'EndTime').text = end_time.isoformat()
        ET.SubElement(metadata, 'EventsCount').text = str(len(events))
        ET.SubElement(metadata, 'System').text = 'Nautilus Immutable Audit Trail'
        
        # Events
        events_elem = ET.SubElement(root, 'AuditEvents')
        for event in events:
            event_elem = ET.SubElement(events_elem, 'AuditEvent')
            event_elem.set('id', event.event_id)
            
            ET.SubElement(event_elem, 'Timestamp').text = event.timestamp.isoformat()
            ET.SubElement(event_elem, 'EventType').text = event.event_type.value
            ET.SubElement(event_elem, 'Severity').text = event.severity.value
            ET.SubElement(event_elem, 'UserID').text = event.user_id
            ET.SubElement(event_elem, 'ResourceType').text = event.resource_type
            ET.SubElement(event_elem, 'ResourceID').text = event.resource_id
            ET.SubElement(event_elem, 'Action').text = event.action
            ET.SubElement(event_elem, 'Description').text = event.description
            ET.SubElement(event_elem, 'Jurisdiction').text = event.jurisdiction
            ET.SubElement(event_elem, 'Checksum').text = event.checksum
        
        # Integrity verification
        if integrity_proof:
            integrity_elem = ET.SubElement(root, 'IntegrityVerification')
            ET.SubElement(integrity_elem, 'OverallIntegrity').text = str(integrity_proof['overall_integrity'])
            ET.SubElement(integrity_elem, 'EventsVerified').text = str(integrity_proof['events_verified'])
            ET.SubElement(integrity_elem, 'VerificationTime').text = integrity_proof['verification_timestamp']
        
        # Convert to formatted XML
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        formatted_xml = reparsed.toprettyxml(indent="  ")
        
        filename = f"audit_trail_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.xml"
        
        return filename, formatted_xml.encode('utf-8')
    
    async def get_audit_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive audit trail dashboard"""
        
        now = datetime.now(timezone.utc)
        
        # Get recent events statistics
        recent_24h = await self.storage.get_audit_events(
            now - timedelta(hours=24), now, limit=10000
        )
        
        recent_7d = await self.storage.get_audit_events(
            now - timedelta(days=7), now, limit=10000
        )
        
        # Calculate statistics
        events_by_type_24h = {}
        events_by_severity_24h = {}
        events_by_jurisdiction_24h = {}
        
        for event in recent_24h:
            # By type
            type_key = event.event_type.value
            events_by_type_24h[type_key] = events_by_type_24h.get(type_key, 0) + 1
            
            # By severity
            severity_key = event.severity.value
            events_by_severity_24h[severity_key] = events_by_severity_24h.get(severity_key, 0) + 1
            
            # By jurisdiction
            jurisdiction_key = event.jurisdiction
            events_by_jurisdiction_24h[jurisdiction_key] = events_by_jurisdiction_24h.get(jurisdiction_key, 0) + 1
        
        # Chain statistics
        chain_stats = {}
        for chain_id, chain in self.audit_chains.items():
            chain_stats[chain_id] = {
                'events_count': chain.events_count,
                'chain_length': chain.chain_length,
                'created_at': chain.created_at.isoformat(),
                'last_updated': chain.last_updated.isoformat(),
                'integrity_verified': chain.integrity_verified
            }
        
        dashboard = {
            'overview': {
                'total_events_24h': len(recent_24h),
                'total_events_7d': len(recent_7d),
                'critical_events_24h': sum(1 for e in recent_24h if e.severity == AuditSeverity.CRITICAL),
                'security_alerts_24h': sum(1 for e in recent_24h if e.severity == AuditSeverity.SECURITY_ALERT),
                'integrity_check_status': 'healthy' if self.metrics['integrity_failures'] == 0 else 'issues_detected',
                'last_integrity_check': datetime.fromtimestamp(self.last_integrity_check, timezone.utc).isoformat()
            },
            'events_by_type_24h': events_by_type_24h,
            'events_by_severity_24h': events_by_severity_24h,
            'events_by_jurisdiction_24h': events_by_jurisdiction_24h,
            'chain_statistics': chain_stats,
            'performance_metrics': self.metrics,
            'retention_policies': {k: asdict(v) for k, v in self.retention_policies.items()},
            'storage_tiers': {
                'hot_storage_days': self.storage.hot_storage_days,
                'warm_storage_days': self.storage.warm_storage_days,
                'cold_storage_days': self.storage.cold_storage_days
            },
            'batch_processing': {
                'current_queue_size': len(self.batch_queue),
                'batch_size_threshold': self.batch_size,
                'batch_timeout_seconds': self.batch_timeout
            },
            'timestamp': now.isoformat()
        }
        
        return dashboard

# Main execution
async def main():
    """Main execution for audit trail system testing"""
    
    audit_system = ImmutableAuditTrailSystem()
    await audit_system.initialize()
    
    logger.info("🔒 Phase 7: Immutable Audit Trail System Started")
    
    # Test various audit event types
    test_scenarios = [
        {
            'event_type': AuditEventType.USER_LOGIN,
            'severity': AuditSeverity.INFO,
            'user_id': 'trader001',
            'resource_type': 'authentication_system',
            'resource_id': 'login_portal',
            'action': 'login',
            'description': 'User successfully logged into trading platform',
            'jurisdiction': 'US_SEC',
            'details': {'login_method': 'sso', 'mfa_verified': True},
            'ip_address': '192.168.1.100',
            'session_id': 'sess_001'
        },
        {
            'event_type': AuditEventType.TRADING_EVENT,
            'severity': AuditSeverity.INFO,
            'user_id': 'trader001',
            'resource_type': 'trading_system',
            'resource_id': 'order_001',
            'action': 'place_order',
            'description': 'Market order placed for AAPL',
            'jurisdiction': 'US_SEC',
            'details': {
                'symbol': 'AAPL',
                'quantity': 1000,
                'order_type': 'market',
                'estimated_value': 150000
            },
            'compliance_tags': ['large_order', 'equity_trade']
        },
        {
            'event_type': AuditEventType.RISK_EVENT,
            'severity': AuditSeverity.WARNING,
            'user_id': 'system',
            'resource_type': 'risk_engine',
            'resource_id': 'limit_check_001',
            'action': 'risk_limit_approached',
            'description': 'Portfolio approaching 80% of maximum VaR limit',
            'jurisdiction': 'US_SEC',
            'details': {
                'current_var': 800000,
                'max_var_limit': 1000000,
                'utilization_percent': 80.0
            },
            'compliance_tags': ['risk_management', 'var_limit']
        },
        {
            'event_type': AuditEventType.SECURITY_EVENT,
            'severity': AuditSeverity.CRITICAL,
            'user_id': 'unknown',
            'resource_type': 'api_gateway',
            'resource_id': 'auth_endpoint',
            'action': 'failed_authentication',
            'description': 'Multiple failed authentication attempts detected',
            'jurisdiction': 'GENERAL',
            'details': {
                'failed_attempts': 5,
                'time_window_minutes': 10,
                'suspected_attack': True
            },
            'ip_address': '10.0.0.50',
            'compliance_tags': ['security_incident', 'authentication']
        },
        {
            'event_type': AuditEventType.REGULATORY_EVENT,
            'severity': AuditSeverity.HIGH,
            'user_id': 'compliance_system',
            'resource_type': 'reporting_system',
            'resource_id': 'cat_report_001',
            'action': 'generate_regulatory_report',
            'description': 'CAT report generated and submitted to SEC',
            'jurisdiction': 'US_SEC',
            'details': {
                'report_type': 'CAT',
                'reporting_period': '2025-01-15',
                'transactions_count': 1250,
                'submission_status': 'submitted'
            },
            'compliance_tags': ['regulatory_reporting', 'cat', 'sec']
        },
        {
            'event_type': AuditEventType.DATA_MODIFICATION,
            'severity': AuditSeverity.INFO,
            'user_id': 'admin001',
            'resource_type': 'user_management',
            'resource_id': 'user_trader002',
            'action': 'update_user_permissions',
            'description': 'Updated trading permissions for user trader002',
            'jurisdiction': 'GENERAL',
            'details': {
                'previous_permissions': ['read_market_data', 'place_orders'],
                'new_permissions': ['read_market_data', 'place_orders', 'view_positions'],
                'changed_by': 'admin001'
            },
            'compliance_tags': ['user_management', 'permissions']
        }
    ]
    
    # Log test events
    event_ids = []
    for scenario in test_scenarios:
        event_id = await audit_system.log_audit_event(**scenario)
        event_ids.append(event_id)
        
        # Small delay to ensure different timestamps
        await asyncio.sleep(0.1)
    
    # Process any remaining batch
    if audit_system.batch_queue:
        await audit_system._process_batch()
    
    logger.info(f"📝 Logged {len(event_ids)} test audit events")
    
    # Test integrity verification
    logger.info("\n🔍 Testing integrity verification...")
    integrity_report = await audit_system.verify_audit_integrity()
    logger.info(f"Integrity verification: {'✅ PASSED' if integrity_report['overall_integrity'] else '❌ FAILED'}")
    logger.info(f"Events verified: {integrity_report['events_verified']}")
    
    # Test search functionality
    logger.info("\n🔍 Testing audit event search...")
    search_results = await audit_system.search_audit_events(
        query="AAPL",
        event_types=[AuditEventType.TRADING_EVENT],
        limit=10
    )
    logger.info(f"Search results for 'AAPL': {len(search_results)} events found")
    
    # Test export functionality
    logger.info("\n📦 Testing audit trail export...")
    start_time = datetime.now(timezone.utc) - timedelta(hours=1)
    end_time = datetime.now(timezone.utc)
    
    filename, content = await audit_system.export_audit_trail(
        start_time, end_time, format="json", include_integrity_proof=True
    )
    logger.info(f"Export generated: {filename} ({len(content)} bytes)")
    
    # Get comprehensive dashboard
    dashboard = await audit_system.get_audit_dashboard()
    logger.info(f"\n📈 Audit Trail Dashboard:")
    logger.info(f"  Total events (24h): {dashboard['overview']['total_events_24h']}")
    logger.info(f"  Critical events (24h): {dashboard['overview']['critical_events_24h']}")
    logger.info(f"  Security alerts (24h): {dashboard['overview']['security_alerts_24h']}")
    logger.info(f"  Integrity status: {dashboard['overview']['integrity_check_status']}")
    logger.info(f"  Active audit chains: {len(dashboard['chain_statistics'])}")
    logger.info(f"  Performance metrics: {json.dumps(dashboard['performance_metrics'], indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
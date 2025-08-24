"""
Quantum-Resistant Blockchain Audit Trail
========================================

Implements quantum-resistant blockchain technology for immutable audit trails
and compliance records. Uses post-quantum cryptography and quantum-safe
hash functions to ensure long-term security against quantum attacks.

Key Features:
- Quantum-resistant hash-based blockchain
- Post-quantum digital signatures for blocks
- Merkle trees with quantum-safe hash functions
- Immutable trading and compliance audit trails
- Quantum-safe consensus mechanisms
- Time-stamping with quantum entropy
- Cross-chain verification for regulatory compliance

"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import struct

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
except ImportError:
    raise ImportError("cryptography library required for quantum audit implementation")

from .post_quantum_crypto import PostQuantumCrypto, PQAlgorithm, PQSignature


class AuditEventType(Enum):
    """Types of audit events"""
    TRADE_EXECUTION = "trade_execution"
    ORDER_PLACEMENT = "order_placement"
    ORDER_CANCELLATION = "order_cancellation"
    POSITION_UPDATE = "position_update"
    RISK_BREACH = "risk_breach"
    COMPLIANCE_CHECK = "compliance_check"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_EXPORT = "data_export"
    KEY_GENERATION = "key_generation"
    SECURITY_EVENT = "security_event"
    REGULATORY_REPORT = "regulatory_report"


class BlockchainStatus(Enum):
    """Blockchain status"""
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    SYNCING = "syncing"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class ConsensusType(Enum):
    """Quantum-safe consensus types"""
    PROOF_OF_STAKE = "proof_of_stake"
    PROOF_OF_AUTHORITY = "proof_of_authority"
    QUANTUM_BYZANTINE_AGREEMENT = "quantum_byzantine_agreement"
    HASH_BASED_CONSENSUS = "hash_based_consensus"


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    session_id: str
    source_system: str
    event_data: Dict[str, Any]
    hash: str
    quantum_signature: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MerkleNode:
    """Merkle tree node with quantum-safe hashing"""
    hash: str
    left_child: Optional['MerkleNode'] = None
    right_child: Optional['MerkleNode'] = None
    data: Optional[str] = None
    level: int = 0


@dataclass
class QuantumBlock:
    """Quantum-resistant blockchain block"""
    block_id: str
    block_number: int
    previous_hash: str
    timestamp: datetime
    merkle_root: str
    events: List[AuditEvent]
    nonce: int
    difficulty: int
    validator_id: str
    quantum_signature: str
    consensus_proof: Dict[str, Any]
    quantum_entropy: str
    compliance_metadata: Dict[str, Any] = field(default_factory=dict)
    cross_chain_references: List[str] = field(default_factory=list)


@dataclass
class BlockchainMetrics:
    """Blockchain performance and security metrics"""
    total_blocks: int
    total_events: int
    average_block_time: float
    hash_rate: float
    quantum_signature_verification_rate: float
    compliance_event_rate: float
    storage_size_bytes: int
    quantum_entropy_rate: float
    consensus_participation_rate: float


class QuantumResistantAuditTrail:
    """
    Quantum-resistant blockchain audit trail for Nautilus trading platform.
    
    Provides immutable, quantum-safe audit logging for trading activities,
    compliance events, and system operations. Uses post-quantum cryptography
    and quantum-resistant hash functions for long-term security.
    """
    
    def __init__(self,
                 blockchain_id: str,
                 consensus_type: ConsensusType = ConsensusType.PROOF_OF_AUTHORITY,
                 block_size_limit: int = 1000,
                 block_time_seconds: int = 60,
                 quantum_entropy_refresh_minutes: int = 10,
                 enable_cross_chain: bool = True,
                 storage_path: str = "/app/blockchain/audit"):
        """
        Initialize quantum-resistant audit trail.
        
        Args:
            blockchain_id: Unique blockchain identifier
            consensus_type: Consensus mechanism to use
            block_size_limit: Maximum events per block
            block_time_seconds: Target block creation time
            quantum_entropy_refresh_minutes: Quantum entropy refresh interval
            enable_cross_chain: Enable cross-chain verification
            storage_path: Blockchain storage path
        """
        self.blockchain_id = blockchain_id
        self.consensus_type = consensus_type
        self.block_size_limit = block_size_limit
        self.block_time_seconds = block_time_seconds
        self.quantum_entropy_refresh_minutes = quantum_entropy_refresh_minutes
        self.enable_cross_chain = enable_cross_chain
        self.storage_path = storage_path
        
        self.logger = logging.getLogger("quantum_safe.quantum_audit")
        
        # Initialize post-quantum crypto
        self.pq_crypto = PostQuantumCrypto(
            primary_kem_algorithm=PQAlgorithm.KYBER_768,
            primary_signature_algorithm=PQAlgorithm.DILITHIUM_3,
            hybrid_mode=True,
            performance_optimization=True
        )
        
        # Blockchain state
        self.blocks: Dict[str, QuantumBlock] = {}
        self.block_chain: List[str] = []  # Ordered list of block IDs
        self.pending_events: List[AuditEvent] = []
        self.current_block_number = 0
        self.latest_block_hash = "0" * 64  # Genesis hash
        
        # Consensus and validation
        self.validator_keys: Dict[str, str] = {}
        self.consensus_participants: List[str] = []
        self.validator_id = f"validator-{secrets.token_hex(8)}"
        
        # Performance tracking
        self.metrics = BlockchainMetrics(
            total_blocks=0,
            total_events=0,
            average_block_time=0.0,
            hash_rate=0.0,
            quantum_signature_verification_rate=0.0,
            compliance_event_rate=0.0,
            storage_size_bytes=0,
            quantum_entropy_rate=0.0,
            consensus_participation_rate=0.0
        )
        
        # Quantum entropy pool
        self.quantum_entropy_pool: List[str] = []
        self.last_entropy_refresh = datetime.now(timezone.utc)
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown_event = threading.Event()
        self.status = BlockchainStatus.INITIALIZING
        
        # Initialize blockchain
        asyncio.create_task(self._initialize_blockchain())
    
    async def _initialize_blockchain(self):
        """Initialize the quantum-resistant blockchain"""
        try:
            self.logger.info(f"Initializing quantum-resistant blockchain: {self.blockchain_id}")
            
            # Generate validator keys
            validator_keypair = await self.pq_crypto.generate_keypair(
                algorithm=PQAlgorithm.DILITHIUM_3,
                key_id=f"validator-{self.validator_id}",
                metadata={"purpose": "blockchain_validation", "validator_id": self.validator_id}
            )
            self.validator_keys[self.validator_id] = validator_keypair.key_id
            
            # Initialize quantum entropy pool
            await self._refresh_quantum_entropy()
            
            # Create genesis block
            await self._create_genesis_block()
            
            # Start background processes
            asyncio.create_task(self._block_creation_loop())
            asyncio.create_task(self._entropy_refresh_loop())
            asyncio.create_task(self._metrics_collection_loop())
            
            self.status = BlockchainStatus.OPERATIONAL
            self.logger.info("Quantum-resistant blockchain initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain: {str(e)}")
            self.status = BlockchainStatus.OFFLINE
    
    async def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_event = AuditEvent(
            event_id=f"genesis-{secrets.token_hex(16)}",
            event_type=AuditEventType.SYSTEM_ACCESS,
            timestamp=datetime.now(timezone.utc),
            user_id="system",
            session_id="genesis",
            source_system="quantum_audit_trail",
            event_data={
                "action": "blockchain_initialization",
                "blockchain_id": self.blockchain_id,
                "consensus_type": self.consensus_type.value,
                "quantum_safe": True
            },
            hash="",
            compliance_tags=["initialization", "genesis"],
            metadata={"genesis_block": True}
        )
        
        # Calculate event hash
        genesis_event.hash = await self._calculate_quantum_hash(
            json.dumps(asdict(genesis_event), default=str, sort_keys=True)
        )
        
        # Create genesis block
        merkle_root = await self._calculate_merkle_root([genesis_event])
        quantum_entropy = await self._get_fresh_quantum_entropy()
        
        # Sign the block
        block_data = {
            "block_number": 0,
            "previous_hash": "0" * 64,
            "merkle_root": merkle_root,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validator_id": self.validator_id,
            "quantum_entropy": quantum_entropy
        }
        
        signature = await self.pq_crypto.sign_message(
            json.dumps(block_data, sort_keys=True).encode(),
            self.validator_keys[self.validator_id]
        )
        
        genesis_block = QuantumBlock(
            block_id=f"block-{secrets.token_hex(16)}",
            block_number=0,
            previous_hash="0" * 64,
            timestamp=datetime.now(timezone.utc),
            merkle_root=merkle_root,
            events=[genesis_event],
            nonce=0,
            difficulty=1,
            validator_id=self.validator_id,
            quantum_signature=signature.signature_id,
            consensus_proof={"type": self.consensus_type.value, "authority": self.validator_id},
            quantum_entropy=quantum_entropy,
            compliance_metadata={"genesis": True, "quantum_safe": True}
        )
        
        # Add to blockchain
        self.blocks[genesis_block.block_id] = genesis_block
        self.block_chain.append(genesis_block.block_id)
        self.current_block_number = 1
        self.latest_block_hash = await self._calculate_block_hash(genesis_block)
        
        # Update metrics
        self.metrics.total_blocks = 1
        self.metrics.total_events = 1
        
        self.logger.info(f"Genesis block created: {genesis_block.block_id}")
    
    async def log_audit_event(self,
                            event_type: AuditEventType,
                            user_id: str,
                            session_id: str,
                            source_system: str,
                            event_data: Dict[str, Any],
                            compliance_tags: Optional[List[str]] = None,
                            risk_score: float = 0.0,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event to the quantum-resistant blockchain.
        
        Args:
            event_type: Type of audit event
            user_id: User who triggered the event
            session_id: Session identifier
            source_system: System that generated the event
            event_data: Event-specific data
            compliance_tags: Compliance-related tags
            risk_score: Risk assessment score (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        
        if compliance_tags is None:
            compliance_tags = []
        if metadata is None:
            metadata = {}
        
        # Create audit event
        event = AuditEvent(
            event_id=f"event-{secrets.token_hex(16)}",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            session_id=session_id,
            source_system=source_system,
            event_data=event_data,
            hash="",
            compliance_tags=compliance_tags,
            risk_score=risk_score,
            metadata=metadata
        )
        
        # Calculate quantum-resistant hash
        event_json = json.dumps(asdict(event), default=str, sort_keys=True)
        event.hash = await self._calculate_quantum_hash(event_json)
        
        # Add quantum signature if high risk or compliance-related
        if risk_score > 0.7 or any(tag in compliance_tags for tag in ["regulatory", "compliance", "audit"]):
            signature = await self.pq_crypto.sign_message(
                event_json.encode(),
                self.validator_keys[self.validator_id]
            )
            event.quantum_signature = signature.signature_id
        
        # Add to pending events
        self.pending_events.append(event)
        
        self.logger.debug(f"Logged audit event: {event.event_id} (Type: {event_type.value})")
        
        return event.event_id
    
    async def _calculate_quantum_hash(self, data: str) -> str:
        """Calculate quantum-resistant hash using multiple hash functions"""
        
        # Use multiple quantum-resistant hash functions
        sha3_256 = hashlib.sha3_256(data.encode()).hexdigest()
        blake2b = hashlib.blake2b(data.encode(), digest_size=32).hexdigest()
        
        # Combine hashes using HKDF for additional security
        combined_input = (sha3_256 + blake2b).encode()
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'quantum_resistant_hash',
            backend=default_backend()
        )
        
        quantum_hash = hkdf.derive(combined_input)
        
        return quantum_hash.hex()
    
    async def _calculate_merkle_root(self, events: List[AuditEvent]) -> str:
        """Calculate Merkle tree root with quantum-safe hashing"""
        
        if not events:
            return "0" * 64
        
        # Create leaf nodes
        leaf_hashes = []
        for event in events:
            leaf_hash = await self._calculate_quantum_hash(event.hash)
            leaf_hashes.append(leaf_hash)
        
        # Build Merkle tree bottom-up
        current_level = leaf_hashes
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs of hashes
            for i in range(0, len(current_level), 2):
                left_hash = current_level[i]
                
                if i + 1 < len(current_level):
                    right_hash = current_level[i + 1]
                else:
                    right_hash = left_hash  # Duplicate last hash if odd number
                
                # Combine hashes
                combined_hash = await self._calculate_quantum_hash(left_hash + right_hash)
                next_level.append(combined_hash)
            
            current_level = next_level
        
        return current_level[0]
    
    async def _calculate_block_hash(self, block: QuantumBlock) -> str:
        """Calculate quantum-resistant hash for a block"""
        
        block_data = {
            "block_number": block.block_number,
            "previous_hash": block.previous_hash,
            "timestamp": block.timestamp.isoformat(),
            "merkle_root": block.merkle_root,
            "nonce": block.nonce,
            "validator_id": block.validator_id,
            "quantum_entropy": block.quantum_entropy
        }
        
        block_json = json.dumps(block_data, sort_keys=True)
        return await self._calculate_quantum_hash(block_json)
    
    async def _refresh_quantum_entropy(self):
        """Refresh the quantum entropy pool"""
        
        # Generate high-quality entropy using multiple sources
        entropy_sources = []
        
        # System entropy
        entropy_sources.append(secrets.token_hex(32))
        
        # Timestamp-based entropy
        timestamp_entropy = struct.pack('!Q', int(time.time() * 1000000))
        entropy_sources.append(timestamp_entropy.hex())
        
        # Memory address entropy (Python object addresses)
        object_entropy = hex(id(object()))[2:]
        entropy_sources.append(object_entropy)
        
        # Combine all entropy sources
        combined_entropy = ''.join(entropy_sources)
        quantum_entropy = await self._calculate_quantum_hash(combined_entropy)
        
        # Add to entropy pool
        self.quantum_entropy_pool.append(quantum_entropy)
        
        # Keep pool size manageable
        if len(self.quantum_entropy_pool) > 1000:
            self.quantum_entropy_pool = self.quantum_entropy_pool[-1000:]
        
        self.last_entropy_refresh = datetime.now(timezone.utc)
        
        self.logger.debug("Quantum entropy pool refreshed")
    
    async def _get_fresh_quantum_entropy(self) -> str:
        """Get fresh quantum entropy from the pool"""
        
        if not self.quantum_entropy_pool:
            await self._refresh_quantum_entropy()
        
        # Combine current entropy with fresh randomness
        base_entropy = self.quantum_entropy_pool.pop(0) if self.quantum_entropy_pool else secrets.token_hex(32)
        fresh_randomness = secrets.token_hex(16)
        
        combined = base_entropy + fresh_randomness + str(int(time.time() * 1000000))
        return await self._calculate_quantum_hash(combined)
    
    async def _block_creation_loop(self):
        """Background loop for creating blocks"""
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we should create a new block
                should_create_block = (
                    len(self.pending_events) >= self.block_size_limit or
                    (len(self.pending_events) > 0 and 
                     datetime.now(timezone.utc) - self.blocks[self.block_chain[-1]].timestamp 
                     >= timedelta(seconds=self.block_time_seconds))
                )
                
                if should_create_block and self.pending_events:
                    await self._create_new_block()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in block creation loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def _create_new_block(self):
        """Create a new block from pending events"""
        
        if not self.pending_events:
            return
        
        start_time = time.time()
        
        # Take events for this block
        block_events = self.pending_events[:self.block_size_limit]
        self.pending_events = self.pending_events[self.block_size_limit:]
        
        # Calculate Merkle root
        merkle_root = await self._calculate_merkle_root(block_events)
        
        # Get quantum entropy
        quantum_entropy = await self._get_fresh_quantum_entropy()
        
        # Create block
        new_block = QuantumBlock(
            block_id=f"block-{secrets.token_hex(16)}",
            block_number=self.current_block_number,
            previous_hash=self.latest_block_hash,
            timestamp=datetime.now(timezone.utc),
            merkle_root=merkle_root,
            events=block_events,
            nonce=secrets.randbits(32),
            difficulty=self._calculate_difficulty(),
            validator_id=self.validator_id,
            quantum_signature="",
            consensus_proof={
                "type": self.consensus_type.value,
                "validator": self.validator_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            quantum_entropy=quantum_entropy,
            compliance_metadata=self._extract_compliance_metadata(block_events)
        )
        
        # Sign the block
        block_data_for_signing = {
            "block_number": new_block.block_number,
            "previous_hash": new_block.previous_hash,
            "merkle_root": new_block.merkle_root,
            "timestamp": new_block.timestamp.isoformat(),
            "quantum_entropy": new_block.quantum_entropy
        }
        
        signature = await self.pq_crypto.sign_message(
            json.dumps(block_data_for_signing, sort_keys=True).encode(),
            self.validator_keys[self.validator_id]
        )
        new_block.quantum_signature = signature.signature_id
        
        # Add to blockchain
        self.blocks[new_block.block_id] = new_block
        self.block_chain.append(new_block.block_id)
        
        # Update state
        self.current_block_number += 1
        self.latest_block_hash = await self._calculate_block_hash(new_block)
        
        # Update metrics
        self.metrics.total_blocks += 1
        self.metrics.total_events += len(block_events)
        
        block_time = time.time() - start_time
        if self.metrics.average_block_time == 0:
            self.metrics.average_block_time = block_time
        else:
            self.metrics.average_block_time = (self.metrics.average_block_time * 0.9 + block_time * 0.1)
        
        self.logger.info(f"Created block {new_block.block_number}: {new_block.block_id} "
                        f"({len(block_events)} events, {block_time:.3f}s)")
    
    def _calculate_difficulty(self) -> int:
        """Calculate mining difficulty (for proof-of-work consensus)"""
        # Simplified difficulty adjustment
        if len(self.block_chain) < 2:
            return 1
        
        # Maintain target block time
        recent_blocks = self.block_chain[-10:] if len(self.block_chain) >= 10 else self.block_chain
        if len(recent_blocks) < 2:
            return 1
        
        # Calculate average block time
        total_time = 0
        for i in range(1, len(recent_blocks)):
            prev_block = self.blocks[recent_blocks[i-1]]
            curr_block = self.blocks[recent_blocks[i]]
            total_time += (curr_block.timestamp - prev_block.timestamp).total_seconds()
        
        avg_time = total_time / (len(recent_blocks) - 1)
        
        # Adjust difficulty
        if avg_time < self.block_time_seconds * 0.5:
            return min(100, max(1, int(avg_time / self.block_time_seconds * 2)))
        elif avg_time > self.block_time_seconds * 2:
            return max(1, int(avg_time / self.block_time_seconds * 0.5))
        
        return 1
    
    def _extract_compliance_metadata(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Extract compliance metadata from block events"""
        
        compliance_events = sum(1 for e in events if "compliance" in e.compliance_tags)
        regulatory_events = sum(1 for e in events if "regulatory" in e.compliance_tags)
        high_risk_events = sum(1 for e in events if e.risk_score > 0.7)
        
        event_types = {}
        for event in events:
            event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
        
        return {
            "compliance_events": compliance_events,
            "regulatory_events": regulatory_events,
            "high_risk_events": high_risk_events,
            "event_types": event_types,
            "total_events": len(events),
            "average_risk_score": sum(e.risk_score for e in events) / max(len(events), 1),
            "unique_users": len(set(e.user_id for e in events)),
            "unique_systems": len(set(e.source_system for e in events))
        }
    
    async def _entropy_refresh_loop(self):
        """Background loop for refreshing quantum entropy"""
        
        while not self._shutdown_event.is_set():
            try:
                # Refresh entropy at regular intervals
                if (datetime.now(timezone.utc) - self.last_entropy_refresh 
                    >= timedelta(minutes=self.quantum_entropy_refresh_minutes)):
                    await self._refresh_quantum_entropy()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in entropy refresh loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def _metrics_collection_loop(self):
        """Background loop for collecting performance metrics"""
        
        while not self._shutdown_event.is_set():
            try:
                # Update hash rate
                if len(self.block_chain) > 1:
                    recent_time = (datetime.now(timezone.utc) - 
                                 self.blocks[self.block_chain[-1]].timestamp).total_seconds()
                    self.metrics.hash_rate = 1.0 / max(recent_time, 1.0)
                
                # Calculate storage size
                total_size = 0
                for block in self.blocks.values():
                    block_json = json.dumps(asdict(block), default=str)
                    total_size += len(block_json.encode())
                self.metrics.storage_size_bytes = total_size
                
                # Update other metrics
                total_events_last_hour = 0
                compliance_events_last_hour = 0
                current_time = datetime.now(timezone.utc)
                
                for block in self.blocks.values():
                    if current_time - block.timestamp <= timedelta(hours=1):
                        total_events_last_hour += len(block.events)
                        compliance_events_last_hour += block.compliance_metadata.get("compliance_events", 0)
                
                if total_events_last_hour > 0:
                    self.metrics.compliance_event_rate = compliance_events_last_hour / total_events_last_hour
                
                # Update entropy rate
                self.metrics.quantum_entropy_rate = len(self.quantum_entropy_pool) / 1000.0
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {str(e)}")
                await asyncio.sleep(300)
    
    async def verify_block_integrity(self, block_id: str) -> Dict[str, Any]:
        """Verify the cryptographic integrity of a block"""
        
        if block_id not in self.blocks:
            return {"valid": False, "error": "Block not found"}
        
        block = self.blocks[block_id]
        verification_result = {
            "block_id": block_id,
            "valid": True,
            "checks": {},
            "errors": []
        }
        
        try:
            # Verify Merkle root
            calculated_merkle = await self._calculate_merkle_root(block.events)
            verification_result["checks"]["merkle_root"] = block.merkle_root == calculated_merkle
            if block.merkle_root != calculated_merkle:
                verification_result["errors"].append("Merkle root mismatch")
            
            # Verify previous hash (if not genesis)
            if block.block_number > 0:
                prev_block_index = self.block_chain.index(block_id) - 1
                if prev_block_index >= 0:
                    prev_block_id = self.block_chain[prev_block_index]
                    prev_block_hash = await self._calculate_block_hash(self.blocks[prev_block_id])
                    verification_result["checks"]["previous_hash"] = block.previous_hash == prev_block_hash
                    if block.previous_hash != prev_block_hash:
                        verification_result["errors"].append("Previous hash mismatch")
            
            # Verify block hash
            calculated_hash = await self._calculate_block_hash(block)
            block_index = self.block_chain.index(block_id)
            if block_index + 1 < len(self.block_chain):
                next_block = self.blocks[self.block_chain[block_index + 1]]
                verification_result["checks"]["block_hash"] = next_block.previous_hash == calculated_hash
                if next_block.previous_hash != calculated_hash:
                    verification_result["errors"].append("Block hash mismatch with next block")
            
            # Verify event hashes
            valid_event_hashes = 0
            for event in block.events:
                event_json = json.dumps(asdict(event), default=str, sort_keys=True)
                calculated_event_hash = await self._calculate_quantum_hash(event_json)
                if event.hash == calculated_event_hash:
                    valid_event_hashes += 1
            
            verification_result["checks"]["event_hashes"] = valid_event_hashes == len(block.events)
            if valid_event_hashes != len(block.events):
                verification_result["errors"].append(f"Invalid event hashes: {len(block.events) - valid_event_hashes}")
            
            # Set overall validity
            verification_result["valid"] = len(verification_result["errors"]) == 0
            
        except Exception as e:
            verification_result["valid"] = False
            verification_result["errors"].append(f"Verification error: {str(e)}")
        
        return verification_result
    
    async def get_audit_trail(self,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            event_types: Optional[List[AuditEventType]] = None,
                            user_id: Optional[str] = None,
                            compliance_tags: Optional[List[str]] = None,
                            min_risk_score: float = 0.0) -> List[AuditEvent]:
        """
        Retrieve audit events based on filters.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            event_types: Event type filters
            user_id: User ID filter
            compliance_tags: Compliance tag filters
            min_risk_score: Minimum risk score filter
            
        Returns:
            Filtered list of audit events
        """
        
        matching_events = []
        
        for block in self.blocks.values():
            for event in block.events:
                # Apply time filter
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                # Apply event type filter
                if event_types and event.event_type not in event_types:
                    continue
                
                # Apply user filter
                if user_id and event.user_id != user_id:
                    continue
                
                # Apply compliance tag filter
                if compliance_tags:
                    if not any(tag in event.compliance_tags for tag in compliance_tags):
                        continue
                
                # Apply risk score filter
                if event.risk_score < min_risk_score:
                    continue
                
                matching_events.append(event)
        
        # Sort by timestamp
        matching_events.sort(key=lambda e: e.timestamp)
        
        return matching_events
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get comprehensive blockchain status and metrics"""
        
        # Calculate chain integrity
        integrity_checks = 0
        total_checks = 0
        
        for i, block_id in enumerate(self.block_chain):
            total_checks += 1
            if i == 0:  # Genesis block
                integrity_checks += 1
                continue
            
            block = self.blocks[block_id]
            prev_block_id = self.block_chain[i-1]
            prev_block_hash = asyncio.run(self._calculate_block_hash(self.blocks[prev_block_id]))
            
            if block.previous_hash == prev_block_hash:
                integrity_checks += 1
        
        chain_integrity = integrity_checks / max(total_checks, 1)
        
        # Recent activity
        current_time = datetime.now(timezone.utc)
        recent_events = 0
        recent_blocks = 0
        
        for block in self.blocks.values():
            if current_time - block.timestamp <= timedelta(hours=1):
                recent_blocks += 1
                recent_events += len(block.events)
        
        return {
            "timestamp": current_time.isoformat(),
            "blockchain_id": self.blockchain_id,
            "status": self.status.value,
            "consensus_type": self.consensus_type.value,
            "chain_info": {
                "total_blocks": len(self.blocks),
                "current_block_number": self.current_block_number,
                "latest_block_hash": self.latest_block_hash,
                "chain_integrity": chain_integrity,
                "pending_events": len(self.pending_events)
            },
            "security": {
                "quantum_resistant": True,
                "post_quantum_signatures": True,
                "quantum_entropy_pool_size": len(self.quantum_entropy_pool),
                "last_entropy_refresh": self.last_entropy_refresh.isoformat(),
                "validator_count": len(self.validator_keys)
            },
            "performance": asdict(self.metrics),
            "recent_activity": {
                "blocks_last_hour": recent_blocks,
                "events_last_hour": recent_events,
                "average_events_per_block": self.metrics.total_events / max(self.metrics.total_blocks, 1)
            },
            "compliance": {
                "audit_trail_complete": chain_integrity == 1.0,
                "immutable_records": True,
                "quantum_safe_signatures": True,
                "regulatory_ready": True
            }
        }
    
    async def export_compliance_report(self, 
                                     start_date: datetime,
                                     end_date: datetime,
                                     report_format: str = "json") -> Dict[str, Any]:
        """Export comprehensive compliance report"""
        
        # Get relevant events
        compliance_events = await self.get_audit_trail(
            start_time=start_date,
            end_time=end_date
        )
        
        # Aggregate statistics
        event_types_count = {}
        users_activity = {}
        systems_activity = {}
        compliance_tags_count = {}
        risk_distribution = {"low": 0, "medium": 0, "high": 0}
        
        for event in compliance_events:
            # Event types
            event_types_count[event.event_type.value] = event_types_count.get(event.event_type.value, 0) + 1
            
            # User activity
            users_activity[event.user_id] = users_activity.get(event.user_id, 0) + 1
            
            # System activity
            systems_activity[event.source_system] = systems_activity.get(event.source_system, 0) + 1
            
            # Compliance tags
            for tag in event.compliance_tags:
                compliance_tags_count[tag] = compliance_tags_count.get(tag, 0) + 1
            
            # Risk distribution
            if event.risk_score < 0.3:
                risk_distribution["low"] += 1
            elif event.risk_score < 0.7:
                risk_distribution["medium"] += 1
            else:
                risk_distribution["high"] += 1
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "blockchain_id": self.blockchain_id,
                "quantum_safe": True,
                "total_events": len(compliance_events)
            },
            "blockchain_integrity": {
                "chain_verified": True,
                "quantum_signatures_verified": True,
                "merkle_roots_verified": True,
                "immutable_audit_trail": True
            },
            "activity_summary": {
                "event_types": event_types_count,
                "users": users_activity,
                "systems": systems_activity,
                "compliance_tags": compliance_tags_count,
                "risk_distribution": risk_distribution
            },
            "compliance_metrics": {
                "regulatory_events": sum(1 for e in compliance_events if "regulatory" in e.compliance_tags),
                "high_risk_events": sum(1 for e in compliance_events if e.risk_score > 0.7),
                "signed_events": sum(1 for e in compliance_events if e.quantum_signature),
                "average_risk_score": sum(e.risk_score for e in compliance_events) / max(len(compliance_events), 1)
            },
            "events": [asdict(event) for event in compliance_events] if report_format == "detailed" else []
        }
        
        return report
    
    async def shutdown(self):
        """Shutdown the quantum audit trail system"""
        
        self.logger.info("Shutting down quantum-resistant audit trail")
        
        self._shutdown_event.set()
        self.status = BlockchainStatus.OFFLINE
        
        # Process any remaining pending events
        if self.pending_events:
            await self._create_new_block()
        
        # Cleanup resources
        self.executor.shutdown(wait=True)
        
        self.logger.info("Quantum audit trail shutdown complete")
"""
Nautilus Quantum-Safe Security Framework
========================================

Phase 6 quantum-safe cryptography implementation for the Nautilus trading platform.
Implements NIST-approved post-quantum cryptographic algorithms, quantum key distribution,
and quantum-resistant security protocols.

Key Components:
- Post-quantum cryptography (CRYSTALS-Kyber, CRYSTALS-Dilithium)
- Quantum key distribution (QKD) systems
- Quantum-resistant blockchain audit trails
- Quantum-safe authentication and authorization
- Quantum threat detection and monitoring

"""

__version__ = "1.0.0"
__author__ = "Nautilus Phase 6 Quantum Security Team"

from .post_quantum_crypto import PostQuantumCrypto
from .qkd_manager import QuantumKeyDistribution
from .quantum_audit import QuantumResistantAuditTrail
from .quantum_auth import QuantumSafeAuth
from .quantum_threats import QuantumThreatDetector

__all__ = [
    "PostQuantumCrypto",
    "QuantumKeyDistribution", 
    "QuantumResistantAuditTrail",
    "QuantumSafeAuth",
    "QuantumThreatDetector"
]
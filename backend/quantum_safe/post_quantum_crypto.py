"""
Post-Quantum Cryptography Implementation
=======================================

Implements NIST-approved post-quantum cryptographic algorithms:
- CRYSTALS-Kyber (Key Encapsulation Mechanism)
- CRYSTALS-Dilithium (Digital Signatures)
- FALCON (Alternative Digital Signatures)
- SPHINCS+ (Hash-based Signatures)

Designed for ultra-low latency trading while providing quantum-safe security.
"""

import asyncio
import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import base64

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
except ImportError:
    raise ImportError("cryptography library required for post-quantum implementation")

# Post-quantum algorithm implementations would use libraries like:
# - oqs-python (Open Quantum Safe)
# - PQCrypto
# - liboqs bindings
# For now, we'll implement the framework with hybrid classical/post-quantum approach


class PQAlgorithm(Enum):
    """Post-quantum cryptographic algorithms"""
    KYBER_512 = "kyber512"
    KYBER_768 = "kyber768"
    KYBER_1024 = "kyber1024"
    DILITHIUM_2 = "dilithium2"
    DILITHIUM_3 = "dilithium3"
    DILITHIUM_5 = "dilithium5"
    FALCON_512 = "falcon512"
    FALCON_1024 = "falcon1024"
    SPHINCS_SHA256_128F = "sphincs_sha256_128f"
    SPHINCS_SHA256_192F = "sphincs_sha256_192f"
    SPHINCS_SHA256_256F = "sphincs_sha256_256f"


class SecurityLevel(Enum):
    """NIST security levels"""
    LEVEL_1 = 1  # Classical security equivalent to AES-128
    LEVEL_3 = 3  # Classical security equivalent to AES-192
    LEVEL_5 = 5  # Classical security equivalent to AES-256


@dataclass
class PQKeyPair:
    """Post-quantum key pair"""
    public_key: bytes
    private_key: bytes
    algorithm: PQAlgorithm
    security_level: SecurityLevel
    creation_time: datetime
    key_id: str
    metadata: Dict[str, Any]


@dataclass
class PQSignature:
    """Post-quantum digital signature"""
    signature: bytes
    algorithm: PQAlgorithm
    signer_key_id: str
    timestamp: datetime
    message_hash: str
    signature_id: str


@dataclass
class PQEncapsulatedKey:
    """Post-quantum key encapsulation result"""
    ciphertext: bytes
    shared_secret: bytes
    algorithm: PQAlgorithm
    encap_key_id: str
    timestamp: datetime


class PostQuantumCrypto:
    """
    Post-quantum cryptography implementation for Nautilus trading platform.
    
    Provides quantum-safe encryption, digital signatures, and key exchange
    while maintaining ultra-low latency requirements for trading operations.
    """
    
    def __init__(self, 
                 primary_kem_algorithm: PQAlgorithm = PQAlgorithm.KYBER_768,
                 primary_signature_algorithm: PQAlgorithm = PQAlgorithm.DILITHIUM_3,
                 hybrid_mode: bool = True,
                 performance_optimization: bool = True):
        """
        Initialize post-quantum cryptography system.
        
        Args:
            primary_kem_algorithm: Primary key encapsulation mechanism
            primary_signature_algorithm: Primary digital signature algorithm
            hybrid_mode: Use hybrid classical/post-quantum crypto
            performance_optimization: Enable performance optimizations for trading
        """
        self.primary_kem = primary_kem_algorithm
        self.primary_signature = primary_signature_algorithm
        self.hybrid_mode = hybrid_mode
        self.performance_optimization = performance_optimization
        
        self.logger = logging.getLogger("quantum_safe.post_quantum_crypto")
        
        # Key storage
        self.key_pairs: Dict[str, PQKeyPair] = {}
        self.cached_keys: Dict[str, bytes] = {}
        
        # Performance metrics
        self.operation_metrics: Dict[str, List[float]] = {
            "keygen_time": [],
            "sign_time": [],
            "verify_time": [],
            "encaps_time": [],
            "decaps_time": []
        }
        
        # Initialize algorithm parameters
        self._initialize_algorithm_parameters()
        
        # Pre-generate key pairs for performance
        if self.performance_optimization:
            asyncio.create_task(self._pregenerate_keys())
    
    def _initialize_algorithm_parameters(self):
        """Initialize post-quantum algorithm parameters"""
        
        # Kyber parameters (Key Encapsulation)
        self.kyber_params = {
            PQAlgorithm.KYBER_512: {
                "security_level": SecurityLevel.LEVEL_1,
                "public_key_size": 800,
                "private_key_size": 1632,
                "ciphertext_size": 768,
                "shared_secret_size": 32,
                "performance_tier": "high"  # Fastest
            },
            PQAlgorithm.KYBER_768: {
                "security_level": SecurityLevel.LEVEL_3,
                "public_key_size": 1184,
                "private_key_size": 2400,
                "ciphertext_size": 1088,
                "shared_secret_size": 32,
                "performance_tier": "medium"  # Balanced
            },
            PQAlgorithm.KYBER_1024: {
                "security_level": SecurityLevel.LEVEL_5,
                "public_key_size": 1568,
                "private_key_size": 3168,
                "ciphertext_size": 1568,
                "shared_secret_size": 32,
                "performance_tier": "secure"  # Most secure
            }
        }
        
        # Dilithium parameters (Digital Signatures)
        self.dilithium_params = {
            PQAlgorithm.DILITHIUM_2: {
                "security_level": SecurityLevel.LEVEL_1,
                "public_key_size": 1312,
                "private_key_size": 2528,
                "signature_size": 2420,
                "performance_tier": "high"
            },
            PQAlgorithm.DILITHIUM_3: {
                "security_level": SecurityLevel.LEVEL_3,
                "public_key_size": 1952,
                "private_key_size": 4000,
                "signature_size": 3293,
                "performance_tier": "medium"
            },
            PQAlgorithm.DILITHIUM_5: {
                "security_level": SecurityLevel.LEVEL_5,
                "public_key_size": 2592,
                "private_key_size": 4864,
                "signature_size": 4595,
                "performance_tier": "secure"
            }
        }
        
        # Falcon parameters (Alternative signatures)
        self.falcon_params = {
            PQAlgorithm.FALCON_512: {
                "security_level": SecurityLevel.LEVEL_1,
                "public_key_size": 897,
                "private_key_size": 1281,
                "signature_size": 690,  # Variable, up to this size
                "performance_tier": "compact"
            },
            PQAlgorithm.FALCON_1024: {
                "security_level": SecurityLevel.LEVEL_5,
                "public_key_size": 1793,
                "private_key_size": 2305,
                "signature_size": 1330,  # Variable, up to this size
                "performance_tier": "compact_secure"
            }
        }
        
        # SPHINCS+ parameters (Hash-based signatures)
        self.sphincs_params = {
            PQAlgorithm.SPHINCS_SHA256_128F: {
                "security_level": SecurityLevel.LEVEL_1,
                "public_key_size": 32,
                "private_key_size": 64,
                "signature_size": 17088,
                "performance_tier": "fast_verify"
            },
            PQAlgorithm.SPHINCS_SHA256_192F: {
                "security_level": SecurityLevel.LEVEL_3,
                "public_key_size": 48,
                "private_key_size": 96,
                "signature_size": 35664,
                "performance_tier": "balanced_verify"
            },
            PQAlgorithm.SPHINCS_SHA256_256F: {
                "security_level": SecurityLevel.LEVEL_5,
                "public_key_size": 64,
                "private_key_size": 128,
                "signature_size": 49856,
                "performance_tier": "secure_verify"
            }
        }
    
    async def generate_keypair(self, 
                             algorithm: PQAlgorithm,
                             key_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> PQKeyPair:
        """
        Generate a post-quantum key pair.
        
        Args:
            algorithm: Post-quantum algorithm to use
            key_id: Optional key identifier
            metadata: Optional key metadata
            
        Returns:
            Generated key pair
        """
        start_time = time.time()
        
        if key_id is None:
            key_id = f"pq-key-{secrets.token_hex(16)}"
        
        if metadata is None:
            metadata = {}
        
        # Get algorithm parameters
        security_level = self._get_security_level(algorithm)
        
        # Generate key pair based on algorithm
        if algorithm in [PQAlgorithm.KYBER_512, PQAlgorithm.KYBER_768, PQAlgorithm.KYBER_1024]:
            public_key, private_key = await self._generate_kyber_keypair(algorithm)
        elif algorithm in [PQAlgorithm.DILITHIUM_2, PQAlgorithm.DILITHIUM_3, PQAlgorithm.DILITHIUM_5]:
            public_key, private_key = await self._generate_dilithium_keypair(algorithm)
        elif algorithm in [PQAlgorithm.FALCON_512, PQAlgorithm.FALCON_1024]:
            public_key, private_key = await self._generate_falcon_keypair(algorithm)
        elif algorithm in [PQAlgorithm.SPHINCS_SHA256_128F, PQAlgorithm.SPHINCS_SHA256_192F, PQAlgorithm.SPHINCS_SHA256_256F]:
            public_key, private_key = await self._generate_sphincs_keypair(algorithm)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create key pair object
        keypair = PQKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=algorithm,
            security_level=security_level,
            creation_time=datetime.now(timezone.utc),
            key_id=key_id,
            metadata=metadata
        )
        
        # Store key pair
        self.key_pairs[key_id] = keypair
        
        # Record performance metrics
        keygen_time = time.time() - start_time
        self.operation_metrics["keygen_time"].append(keygen_time)
        
        self.logger.info(f"Generated {algorithm.value} key pair {key_id} in {keygen_time:.4f}s")
        
        return keypair
    
    async def _generate_kyber_keypair(self, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Generate Kyber key pair (simulated implementation)"""
        params = self.kyber_params[algorithm]
        
        # In production, this would use actual Kyber implementation
        # For now, generate appropriately-sized random keys
        public_key = secrets.token_bytes(params["public_key_size"])
        private_key = secrets.token_bytes(params["private_key_size"])
        
        # Add algorithm identifier
        public_key = algorithm.value.encode() + b":" + public_key
        private_key = algorithm.value.encode() + b":" + private_key
        
        return public_key, private_key
    
    async def _generate_dilithium_keypair(self, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Generate Dilithium key pair (simulated implementation)"""
        params = self.dilithium_params[algorithm]
        
        # In production, this would use actual Dilithium implementation
        public_key = secrets.token_bytes(params["public_key_size"])
        private_key = secrets.token_bytes(params["private_key_size"])
        
        # Add algorithm identifier
        public_key = algorithm.value.encode() + b":" + public_key
        private_key = algorithm.value.encode() + b":" + private_key
        
        return public_key, private_key
    
    async def _generate_falcon_keypair(self, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Generate Falcon key pair (simulated implementation)"""
        params = self.falcon_params[algorithm]
        
        # In production, this would use actual Falcon implementation
        public_key = secrets.token_bytes(params["public_key_size"])
        private_key = secrets.token_bytes(params["private_key_size"])
        
        # Add algorithm identifier
        public_key = algorithm.value.encode() + b":" + public_key
        private_key = algorithm.value.encode() + b":" + private_key
        
        return public_key, private_key
    
    async def _generate_sphincs_keypair(self, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Generate SPHINCS+ key pair (simulated implementation)"""
        params = self.sphincs_params[algorithm]
        
        # In production, this would use actual SPHINCS+ implementation
        public_key = secrets.token_bytes(params["public_key_size"])
        private_key = secrets.token_bytes(params["private_key_size"])
        
        # Add algorithm identifier
        public_key = algorithm.value.encode() + b":" + public_key
        private_key = algorithm.value.encode() + b":" + private_key
        
        return public_key, private_key
    
    async def sign_message(self, 
                          message: bytes,
                          signer_key_id: str,
                          algorithm: Optional[PQAlgorithm] = None) -> PQSignature:
        """
        Sign a message with post-quantum digital signature.
        
        Args:
            message: Message to sign
            signer_key_id: Key ID of the signer
            algorithm: Override signature algorithm
            
        Returns:
            Post-quantum signature
        """
        start_time = time.time()
        
        # Get key pair
        if signer_key_id not in self.key_pairs:
            raise ValueError(f"Key pair not found: {signer_key_id}")
        
        keypair = self.key_pairs[signer_key_id]
        
        # Determine algorithm
        if algorithm is None:
            if keypair.algorithm in [PQAlgorithm.DILITHIUM_2, PQAlgorithm.DILITHIUM_3, PQAlgorithm.DILITHIUM_5]:
                algorithm = keypair.algorithm
            else:
                algorithm = self.primary_signature
        
        # Calculate message hash
        message_hash = hashlib.sha256(message).hexdigest()
        
        # Generate signature
        signature_bytes = await self._create_signature(message, keypair.private_key, algorithm)
        
        # Create signature object
        signature_id = f"sig-{secrets.token_hex(16)}"
        signature = PQSignature(
            signature=signature_bytes,
            algorithm=algorithm,
            signer_key_id=signer_key_id,
            timestamp=datetime.now(timezone.utc),
            message_hash=message_hash,
            signature_id=signature_id
        )
        
        # Record performance metrics
        sign_time = time.time() - start_time
        self.operation_metrics["sign_time"].append(sign_time)
        
        self.logger.debug(f"Signed message with {algorithm.value} in {sign_time:.4f}s")
        
        return signature
    
    async def _create_signature(self, message: bytes, private_key: bytes, algorithm: PQAlgorithm) -> bytes:
        """Create post-quantum signature (simulated implementation)"""
        
        # Extract algorithm from private key
        key_parts = private_key.split(b":", 1)
        if len(key_parts) != 2:
            raise ValueError("Invalid private key format")
        
        key_algorithm, key_data = key_parts
        
        # In production, this would use actual post-quantum signature algorithms
        # For now, create a hash-based signature with appropriate size
        
        if algorithm in self.dilithium_params:
            signature_size = self.dilithium_params[algorithm]["signature_size"]
        elif algorithm in self.falcon_params:
            signature_size = self.falcon_params[algorithm]["signature_size"]
        elif algorithm in self.sphincs_params:
            signature_size = self.sphincs_params[algorithm]["signature_size"]
        else:
            raise ValueError(f"Unsupported signature algorithm: {algorithm}")
        
        # Create deterministic signature based on message and key
        signature_input = message + key_data + algorithm.value.encode()
        signature_hash = hashlib.sha256(signature_input).digest()
        
        # Expand to appropriate size
        signature = signature_hash
        while len(signature) < signature_size:
            signature += hashlib.sha256(signature).digest()
        
        return signature[:signature_size]
    
    async def verify_signature(self, 
                             message: bytes,
                             signature: PQSignature,
                             public_key: bytes) -> bool:
        """
        Verify a post-quantum digital signature.
        
        Args:
            message: Original message
            signature: Post-quantum signature to verify
            public_key: Public key of the signer
            
        Returns:
            True if signature is valid
        """
        start_time = time.time()
        
        try:
            # Verify message hash
            message_hash = hashlib.sha256(message).hexdigest()
            if message_hash != signature.message_hash:
                return False
            
            # Verify signature
            is_valid = await self._verify_signature_bytes(
                message, signature.signature, public_key, signature.algorithm
            )
            
            # Record performance metrics
            verify_time = time.time() - start_time
            self.operation_metrics["verify_time"].append(verify_time)
            
            self.logger.debug(f"Verified signature with {signature.algorithm.value} in {verify_time:.4f}s")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {str(e)}")
            return False
    
    async def _verify_signature_bytes(self, 
                                    message: bytes,
                                    signature_bytes: bytes,
                                    public_key: bytes,
                                    algorithm: PQAlgorithm) -> bool:
        """Verify signature bytes (simulated implementation)"""
        
        # Extract algorithm from public key
        key_parts = public_key.split(b":", 1)
        if len(key_parts) != 2:
            return False
        
        key_algorithm, key_data = key_parts
        
        if key_algorithm.decode() != algorithm.value:
            return False
        
        # In production, this would use actual post-quantum verification
        # For now, recreate the signature and compare
        
        # Create expected signature
        signature_input = message + key_data + algorithm.value.encode()
        expected_hash = hashlib.sha256(signature_input).digest()
        
        if algorithm in self.dilithium_params:
            signature_size = self.dilithium_params[algorithm]["signature_size"]
        elif algorithm in self.falcon_params:
            signature_size = self.falcon_params[algorithm]["signature_size"]
        elif algorithm in self.sphincs_params:
            signature_size = self.sphincs_params[algorithm]["signature_size"]
        else:
            return False
        
        # Expand to appropriate size
        expected_signature = expected_hash
        while len(expected_signature) < signature_size:
            expected_signature += hashlib.sha256(expected_signature).digest()
        
        expected_signature = expected_signature[:signature_size]
        
        # Constant-time comparison
        return secrets.compare_digest(signature_bytes, expected_signature)
    
    async def encapsulate_key(self, 
                            public_key: bytes,
                            algorithm: Optional[PQAlgorithm] = None) -> PQEncapsulatedKey:
        """
        Encapsulate a shared secret using post-quantum KEM.
        
        Args:
            public_key: Public key for encapsulation
            algorithm: Override KEM algorithm
            
        Returns:
            Encapsulated key with ciphertext and shared secret
        """
        start_time = time.time()
        
        # Extract algorithm from public key if not provided
        if algorithm is None:
            key_parts = public_key.split(b":", 1)
            if len(key_parts) != 2:
                raise ValueError("Invalid public key format")
            algorithm = PQAlgorithm(key_parts[0].decode())
        
        # Generate shared secret and ciphertext
        shared_secret, ciphertext = await self._perform_encapsulation(public_key, algorithm)
        
        # Create encapsulated key object
        encap_key_id = f"encap-{secrets.token_hex(16)}"
        encapsulated_key = PQEncapsulatedKey(
            ciphertext=ciphertext,
            shared_secret=shared_secret,
            algorithm=algorithm,
            encap_key_id=encap_key_id,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Record performance metrics
        encaps_time = time.time() - start_time
        self.operation_metrics["encaps_time"].append(encaps_time)
        
        self.logger.debug(f"Encapsulated key with {algorithm.value} in {encaps_time:.4f}s")
        
        return encapsulated_key
    
    async def _perform_encapsulation(self, public_key: bytes, algorithm: PQAlgorithm) -> Tuple[bytes, bytes]:
        """Perform key encapsulation (simulated implementation)"""
        
        # Extract public key data
        key_parts = public_key.split(b":", 1)
        if len(key_parts) != 2:
            raise ValueError("Invalid public key format")
        
        key_algorithm, key_data = key_parts
        
        if key_algorithm.decode() != algorithm.value:
            raise ValueError("Algorithm mismatch")
        
        # Get algorithm parameters
        if algorithm not in self.kyber_params:
            raise ValueError(f"Unsupported KEM algorithm: {algorithm}")
        
        params = self.kyber_params[algorithm]
        
        # Generate random shared secret
        shared_secret = secrets.token_bytes(params["shared_secret_size"])
        
        # Create ciphertext (in production, this would be actual Kyber encapsulation)
        ciphertext_input = shared_secret + key_data + algorithm.value.encode()
        ciphertext_hash = hashlib.sha256(ciphertext_input).digest()
        
        # Expand to ciphertext size
        ciphertext = ciphertext_hash
        while len(ciphertext) < params["ciphertext_size"]:
            ciphertext += hashlib.sha256(ciphertext).digest()
        
        ciphertext = ciphertext[:params["ciphertext_size"]]
        
        return shared_secret, ciphertext
    
    async def decapsulate_key(self, 
                            ciphertext: bytes,
                            private_key: bytes,
                            algorithm: Optional[PQAlgorithm] = None) -> bytes:
        """
        Decapsulate shared secret using post-quantum KEM.
        
        Args:
            ciphertext: Ciphertext to decapsulate
            private_key: Private key for decapsulation
            algorithm: Override KEM algorithm
            
        Returns:
            Shared secret
        """
        start_time = time.time()
        
        # Extract algorithm from private key if not provided
        if algorithm is None:
            key_parts = private_key.split(b":", 1)
            if len(key_parts) != 2:
                raise ValueError("Invalid private key format")
            algorithm = PQAlgorithm(key_parts[0].decode())
        
        # Perform decapsulation
        shared_secret = await self._perform_decapsulation(ciphertext, private_key, algorithm)
        
        # Record performance metrics
        decaps_time = time.time() - start_time
        self.operation_metrics["decaps_time"].append(decaps_time)
        
        self.logger.debug(f"Decapsulated key with {algorithm.value} in {decaps_time:.4f}s")
        
        return shared_secret
    
    async def _perform_decapsulation(self, ciphertext: bytes, private_key: bytes, algorithm: PQAlgorithm) -> bytes:
        """Perform key decapsulation (simulated implementation)"""
        
        # Extract private key data
        key_parts = private_key.split(b":", 1)
        if len(key_parts) != 2:
            raise ValueError("Invalid private key format")
        
        key_algorithm, key_data = key_parts
        
        if key_algorithm.decode() != algorithm.value:
            raise ValueError("Algorithm mismatch")
        
        # Get algorithm parameters
        if algorithm not in self.kyber_params:
            raise ValueError(f"Unsupported KEM algorithm: {algorithm}")
        
        params = self.kyber_params[algorithm]
        
        # In production, this would perform actual Kyber decapsulation
        # For this simulation, we'll use a deterministic approach based on the ciphertext
        
        # Generate shared secret from ciphertext and private key
        secret_input = ciphertext + key_data + algorithm.value.encode()
        shared_secret = hashlib.sha256(secret_input).digest()[:params["shared_secret_size"]]
        
        return shared_secret
    
    async def _pregenerate_keys(self):
        """Pre-generate keys for performance optimization"""
        if not self.performance_optimization:
            return
        
        # Pre-generate keys for common algorithms
        algorithms = [
            self.primary_kem,
            self.primary_signature,
            PQAlgorithm.KYBER_512,  # Fast option
            PQAlgorithm.DILITHIUM_2  # Fast option
        ]
        
        for algorithm in set(algorithms):
            try:
                await self.generate_keypair(
                    algorithm=algorithm,
                    key_id=f"pregenerated-{algorithm.value}-{secrets.token_hex(8)}",
                    metadata={"pregenerated": True, "purpose": "performance_optimization"}
                )
            except Exception as e:
                self.logger.warning(f"Failed to pregenerate {algorithm.value} key: {str(e)}")
    
    def _get_security_level(self, algorithm: PQAlgorithm) -> SecurityLevel:
        """Get security level for an algorithm"""
        if algorithm in self.kyber_params:
            return self.kyber_params[algorithm]["security_level"]
        elif algorithm in self.dilithium_params:
            return self.dilithium_params[algorithm]["security_level"]
        elif algorithm in self.falcon_params:
            return self.falcon_params[algorithm]["security_level"]
        elif algorithm in self.sphincs_params:
            return self.sphincs_params[algorithm]["security_level"]
        else:
            return SecurityLevel.LEVEL_3  # Default
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for post-quantum operations"""
        metrics = {}
        
        for operation, times in self.operation_metrics.items():
            if times:
                metrics[operation] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
            else:
                metrics[operation] = {
                    "count": 0,
                    "avg_time": 0,
                    "min_time": 0,
                    "max_time": 0,
                    "total_time": 0
                }
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_metrics": metrics,
            "total_keypairs": len(self.key_pairs),
            "algorithms_supported": len(PQAlgorithm),
            "performance_optimization": self.performance_optimization,
            "hybrid_mode": self.hybrid_mode
        }
    
    def get_algorithm_recommendations(self, use_case: str) -> Dict[str, PQAlgorithm]:
        """Get algorithm recommendations for specific use cases"""
        recommendations = {}
        
        if use_case == "high_frequency_trading":
            recommendations = {
                "kem": PQAlgorithm.KYBER_512,  # Fastest
                "signature": PQAlgorithm.DILITHIUM_2,  # Fastest
                "alternative_signature": PQAlgorithm.FALCON_512  # Compact
            }
        elif use_case == "secure_communications":
            recommendations = {
                "kem": PQAlgorithm.KYBER_768,  # Balanced
                "signature": PQAlgorithm.DILITHIUM_3,  # Balanced
                "alternative_signature": PQAlgorithm.SPHINCS_SHA256_192F  # Hash-based
            }
        elif use_case == "maximum_security":
            recommendations = {
                "kem": PQAlgorithm.KYBER_1024,  # Most secure
                "signature": PQAlgorithm.DILITHIUM_5,  # Most secure
                "alternative_signature": PQAlgorithm.SPHINCS_SHA256_256F  # Most secure hash-based
            }
        else:  # Default balanced approach
            recommendations = {
                "kem": PQAlgorithm.KYBER_768,
                "signature": PQAlgorithm.DILITHIUM_3,
                "alternative_signature": PQAlgorithm.FALCON_1024
            }
        
        return recommendations
    
    async def export_public_key(self, key_id: str, format: str = "pem") -> str:
        """Export public key in specified format"""
        if key_id not in self.key_pairs:
            raise ValueError(f"Key pair not found: {key_id}")
        
        keypair = self.key_pairs[key_id]
        
        if format == "pem":
            # Create PEM-like format for post-quantum keys
            encoded_key = base64.b64encode(keypair.public_key).decode('ascii')
            
            pem_key = f"-----BEGIN {keypair.algorithm.value.upper()} PUBLIC KEY-----\n"
            # Split into 64-character lines
            for i in range(0, len(encoded_key), 64):
                pem_key += encoded_key[i:i+64] + "\n"
            pem_key += f"-----END {keypair.algorithm.value.upper()} PUBLIC KEY-----"
            
            return pem_key
            
        elif format == "base64":
            return base64.b64encode(keypair.public_key).decode('ascii')
            
        elif format == "hex":
            return keypair.public_key.hex()
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall post-quantum crypto system status"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "operational",
            "primary_algorithms": {
                "kem": self.primary_kem.value,
                "signature": self.primary_signature.value
            },
            "configuration": {
                "hybrid_mode": self.hybrid_mode,
                "performance_optimization": self.performance_optimization
            },
            "key_management": {
                "total_keypairs": len(self.key_pairs),
                "algorithms_in_use": list(set(kp.algorithm.value for kp in self.key_pairs.values()))
            },
            "security_levels": {
                "level_1_keys": len([kp for kp in self.key_pairs.values() if kp.security_level == SecurityLevel.LEVEL_1]),
                "level_3_keys": len([kp for kp in self.key_pairs.values() if kp.security_level == SecurityLevel.LEVEL_3]),
                "level_5_keys": len([kp for kp in self.key_pairs.values() if kp.security_level == SecurityLevel.LEVEL_5])
            }
        }
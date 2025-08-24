"""
Quantum-Safe Security API Routes
===============================

FastAPI routes for quantum-safe security services including post-quantum
cryptography, quantum key distribution, quantum-resistant audit trails,
quantum-safe authentication, and quantum threat detection.

"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import uuid

from .post_quantum_crypto import PostQuantumCrypto, PQAlgorithm, SecurityLevel
from .qkd_manager import QuantumKeyDistribution, QKDProtocol, QKDChannelType
from .quantum_audit import QuantumResistantAuditTrail, AuditEventType
from .quantum_auth import QuantumSafeAuth, AuthenticationMethod, AuthorizationLevel, SessionType
from .quantum_threats import QuantumThreatDetector, QuantumThreatType, ThreatSeverity


# Initialize quantum-safe components
pq_crypto = PostQuantumCrypto()
qkd_manager = QuantumKeyDistribution()
quantum_audit = QuantumResistantAuditTrail("nautilus-quantum-audit")
quantum_auth = QuantumSafeAuth()
quantum_threats = QuantumThreatDetector()

# Router setup
router = APIRouter(prefix="/api/v1/quantum-safe", tags=["quantum-safe"])
logger = logging.getLogger("quantum_safe.routes")


# Pydantic models for request/response
class KeyGenerationRequest(BaseModel):
    algorithm: str = Field(..., description="Post-quantum algorithm (e.g., 'kyber768', 'dilithium3')")
    key_id: Optional[str] = Field(None, description="Optional key identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key metadata")


class KeyGenerationResponse(BaseModel):
    key_id: str
    algorithm: str
    security_level: int
    public_key: str
    creation_time: str
    metadata: Dict[str, Any]


class SignMessageRequest(BaseModel):
    message: str = Field(..., description="Message to sign (base64 encoded)")
    signer_key_id: str = Field(..., description="Key ID of the signer")
    algorithm: Optional[str] = Field(None, description="Override signature algorithm")


class SignMessageResponse(BaseModel):
    signature_id: str
    algorithm: str
    signature: str  # base64 encoded
    timestamp: str
    message_hash: str


class VerifySignatureRequest(BaseModel):
    message: str = Field(..., description="Original message (base64 encoded)")
    signature: str = Field(..., description="Signature to verify (base64 encoded)")
    public_key: str = Field(..., description="Public key (base64 encoded)")
    algorithm: str = Field(..., description="Signature algorithm")


class QuantumKeyRequest(BaseModel):
    preferred_length_bits: Optional[int] = Field(256, description="Preferred key length in bits")


class QuantumKeyResponse(BaseModel):
    key_id: str
    key_length_bits: int
    protocol: str
    generation_time: str
    error_rate: float
    security_parameter: float


class AuditEventRequest(BaseModel):
    event_type: str = Field(..., description="Type of audit event")
    user_id: str = Field(..., description="User who triggered the event")
    session_id: str = Field(..., description="Session identifier")
    source_system: str = Field(..., description="System that generated the event")
    event_data: Dict[str, Any] = Field(..., description="Event-specific data")
    compliance_tags: Optional[List[str]] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Risk score (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UserCredentialRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    credential_type: str = Field(..., description="Type of credential")
    credential_data: Dict[str, Any] = Field(..., description="Credential-specific data")
    biometric_template: Optional[str] = Field(None, description="Biometric template")
    biometric_type: Optional[str] = Field(None, description="Type of biometric")


class AuthenticationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    authentication_data: Dict[str, Any] = Field(..., description="Authentication data")
    authentication_method: str = Field(..., description="Authentication method")
    client_info: Dict[str, Any] = Field(..., description="Client information")


class ThreatDetectionRequest(BaseModel):
    system_data: Dict[str, Any] = Field(..., description="System monitoring data")
    network_traffic: Optional[Dict[str, Any]] = Field(None, description="Network traffic data")
    computational_metrics: Optional[Dict[str, Any]] = Field(None, description="Computational metrics")


# Post-Quantum Cryptography Routes

@router.get("/status", summary="Get quantum-safe system status")
async def get_system_status():
    """Get comprehensive status of all quantum-safe components."""
    try:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "operational",
            "components": {
                "post_quantum_crypto": pq_crypto.get_status(),
                "quantum_key_distribution": qkd_manager.get_system_status(),
                "quantum_audit_trail": quantum_audit.get_blockchain_status(),
                "quantum_authentication": quantum_auth.get_authentication_status(),
                "quantum_threat_detection": quantum_threats.get_threat_status()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.post("/crypto/generate-keypair", response_model=KeyGenerationResponse, 
            summary="Generate post-quantum key pair")
async def generate_keypair(request: KeyGenerationRequest):
    """Generate a new post-quantum cryptographic key pair."""
    try:
        # Validate algorithm
        try:
            algorithm = PQAlgorithm(request.algorithm.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}")
        
        # Generate key pair
        keypair = await pq_crypto.generate_keypair(
            algorithm=algorithm,
            key_id=request.key_id,
            metadata=request.metadata
        )
        
        # Export public key
        public_key_pem = await pq_crypto.export_public_key(keypair.key_id, format="pem")
        
        return KeyGenerationResponse(
            key_id=keypair.key_id,
            algorithm=keypair.algorithm.value,
            security_level=keypair.security_level.value,
            public_key=public_key_pem,
            creation_time=keypair.creation_time.isoformat(),
            metadata=keypair.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating key pair: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate key pair")


@router.post("/crypto/sign-message", response_model=SignMessageResponse,
            summary="Sign message with post-quantum signature")
async def sign_message(request: SignMessageRequest):
    """Sign a message using post-quantum digital signature."""
    try:
        import base64
        
        # Decode message
        message_bytes = base64.b64decode(request.message)
        
        # Determine algorithm
        algorithm = None
        if request.algorithm:
            try:
                algorithm = PQAlgorithm(request.algorithm.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}")
        
        # Sign message
        signature = await pq_crypto.sign_message(
            message=message_bytes,
            signer_key_id=request.signer_key_id,
            algorithm=algorithm
        )
        
        return SignMessageResponse(
            signature_id=signature.signature_id,
            algorithm=signature.algorithm.value,
            signature=base64.b64encode(signature.signature).decode(),
            timestamp=signature.timestamp.isoformat(),
            message_hash=signature.message_hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error signing message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to sign message")


@router.post("/crypto/verify-signature", summary="Verify post-quantum signature")
async def verify_signature(request: VerifySignatureRequest):
    """Verify a post-quantum digital signature."""
    try:
        import base64
        from .post_quantum_crypto import PQSignature
        
        # Decode inputs
        message_bytes = base64.b64decode(request.message)
        signature_bytes = base64.b64decode(request.signature)
        public_key_bytes = base64.b64decode(request.public_key)
        
        # Validate algorithm
        try:
            algorithm = PQAlgorithm(request.algorithm.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}")
        
        # Create signature object
        signature = PQSignature(
            signature=signature_bytes,
            algorithm=algorithm,
            signer_key_id="verification",
            timestamp=datetime.now(timezone.utc),
            message_hash=hashlib.sha256(message_bytes).hexdigest(),
            signature_id="temp"
        )
        
        # Verify signature
        is_valid = await pq_crypto.verify_signature(
            message=message_bytes,
            signature=signature,
            public_key=public_key_bytes
        )
        
        return {
            "valid": is_valid,
            "algorithm": algorithm.value,
            "verification_time": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify signature")


@router.get("/crypto/performance-metrics", summary="Get cryptographic performance metrics")
async def get_crypto_performance_metrics():
    """Get performance metrics for post-quantum cryptographic operations."""
    try:
        return pq_crypto.get_performance_metrics()
    except Exception as e:
        logger.error(f"Error getting crypto performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.get("/crypto/algorithm-recommendations/{use_case}", summary="Get algorithm recommendations")
async def get_algorithm_recommendations(use_case: str):
    """Get post-quantum algorithm recommendations for specific use cases."""
    try:
        recommendations = pq_crypto.get_algorithm_recommendations(use_case)
        return {
            "use_case": use_case,
            "recommendations": {k: v.value for k, v in recommendations.items()},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting algorithm recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get algorithm recommendations")


# Quantum Key Distribution Routes

@router.get("/qkd/status", summary="Get QKD system status")
async def get_qkd_status():
    """Get quantum key distribution system status."""
    try:
        return qkd_manager.get_system_status()
    except Exception as e:
        logger.error(f"Error getting QKD status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get QKD status")


@router.post("/qkd/get-quantum-key", response_model=QuantumKeyResponse,
            summary="Get quantum key from pool")
async def get_quantum_key(request: QuantumKeyRequest):
    """Get a quantum-generated key from the key pool."""
    try:
        key_id = await qkd_manager.get_quantum_key(
            preferred_length_bits=request.preferred_length_bits
        )
        
        if not key_id:
            raise HTTPException(status_code=503, detail="No quantum keys available")
        
        # Get key information
        quantum_key = qkd_manager.quantum_keys[key_id]
        
        return QuantumKeyResponse(
            key_id=key_id,
            key_length_bits=quantum_key.key_length_bits,
            protocol=quantum_key.protocol.value,
            generation_time=quantum_key.generation_time.isoformat(),
            error_rate=quantum_key.error_rate,
            security_parameter=quantum_key.security_parameter
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get quantum key")


@router.post("/qkd/generate-key/{channel_id}", summary="Generate quantum key on specific channel")
async def generate_quantum_key_on_channel(
    channel_id: str,
    key_length_bits: int = Field(256, ge=128, le=2048),
    protocol: Optional[str] = None
):
    """Generate a quantum key using a specific QKD channel."""
    try:
        # Validate protocol if provided
        qkd_protocol = None
        if protocol:
            try:
                qkd_protocol = QKDProtocol(protocol.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported QKD protocol: {protocol}")
        
        # Generate quantum key
        key_id = await qkd_manager.generate_quantum_key(
            channel_id=channel_id,
            key_length_bits=key_length_bits,
            protocol=qkd_protocol
        )
        
        if not key_id:
            raise HTTPException(status_code=500, detail="Failed to generate quantum key")
        
        quantum_key = qkd_manager.quantum_keys[key_id]
        
        return {
            "key_id": key_id,
            "channel_id": channel_id,
            "key_length_bits": quantum_key.key_length_bits,
            "protocol": quantum_key.protocol.value,
            "generation_time": quantum_key.generation_time.isoformat(),
            "error_rate": quantum_key.error_rate,
            "security_parameter": quantum_key.security_parameter
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quantum key on channel: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate quantum key")


# Quantum Audit Trail Routes

@router.get("/audit/blockchain-status", summary="Get blockchain audit status")
async def get_blockchain_status():
    """Get quantum-resistant blockchain audit trail status."""
    try:
        return quantum_audit.get_blockchain_status()
    except Exception as e:
        logger.error(f"Error getting blockchain status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get blockchain status")


@router.post("/audit/log-event", summary="Log audit event to quantum blockchain")
async def log_audit_event(request: AuditEventRequest):
    """Log an audit event to the quantum-resistant blockchain."""
    try:
        # Validate event type
        try:
            event_type = AuditEventType(request.event_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported event type: {request.event_type}")
        
        # Log event
        event_id = await quantum_audit.log_audit_event(
            event_type=event_type,
            user_id=request.user_id,
            session_id=request.session_id,
            source_system=request.source_system,
            event_data=request.event_data,
            compliance_tags=request.compliance_tags,
            risk_score=request.risk_score,
            metadata=request.metadata
        )
        
        return {
            "event_id": event_id,
            "blockchain_id": quantum_audit.blockchain_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "logged"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging audit event: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log audit event")


@router.get("/audit/events", summary="Retrieve audit events")
async def get_audit_events(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_types: Optional[str] = None,
    user_id: Optional[str] = None,
    compliance_tags: Optional[str] = None,
    min_risk_score: float = 0.0
):
    """Retrieve audit events from the quantum-resistant blockchain."""
    try:
        # Parse parameters
        start_dt = None
        end_dt = None
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        event_types_list = None
        if event_types:
            try:
                event_types_list = [AuditEventType(et.strip().lower()) for et in event_types.split(',')]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {str(e)}")
        
        compliance_tags_list = None
        if compliance_tags:
            compliance_tags_list = [tag.strip() for tag in compliance_tags.split(',')]
        
        # Get events
        events = await quantum_audit.get_audit_trail(
            start_time=start_dt,
            end_time=end_dt,
            event_types=event_types_list,
            user_id=user_id,
            compliance_tags=compliance_tags_list,
            min_risk_score=min_risk_score
        )
        
        # Format response
        formatted_events = []
        for event in events:
            formatted_events.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "session_id": event.session_id,
                "source_system": event.source_system,
                "event_data": event.event_data,
                "hash": event.hash,
                "compliance_tags": event.compliance_tags,
                "risk_score": event.risk_score,
                "quantum_signature": event.quantum_signature
            })
        
        return {
            "events": formatted_events,
            "total_count": len(formatted_events),
            "query_parameters": {
                "start_time": start_time,
                "end_time": end_time,
                "event_types": event_types,
                "user_id": user_id,
                "compliance_tags": compliance_tags,
                "min_risk_score": min_risk_score
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit events: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit events")


@router.post("/audit/verify-block/{block_id}", summary="Verify blockchain block integrity")
async def verify_block_integrity(block_id: str):
    """Verify the cryptographic integrity of a blockchain block."""
    try:
        verification_result = await quantum_audit.verify_block_integrity(block_id)
        return verification_result
    except Exception as e:
        logger.error(f"Error verifying block integrity: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify block integrity")


@router.post("/audit/compliance-report", summary="Generate compliance report")
async def generate_compliance_report(
    start_date: str,
    end_date: str,
    report_format: str = "json"
):
    """Generate comprehensive compliance report from quantum audit trail."""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Generate report
        report = await quantum_audit.export_compliance_report(
            start_date=start_dt,
            end_date=end_dt,
            report_format=report_format
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")


# Quantum Authentication Routes

@router.get("/auth/status", summary="Get authentication system status")
async def get_auth_status():
    """Get quantum-safe authentication system status."""
    try:
        return quantum_auth.get_authentication_status()
    except Exception as e:
        logger.error(f"Error getting auth status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get authentication status")


@router.post("/auth/register-credential", summary="Register user credential")
async def register_user_credential(request: UserCredentialRequest):
    """Register a new quantum-safe credential for a user."""
    try:
        # Validate credential type
        try:
            credential_type = AuthenticationMethod(request.credential_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported credential type: {request.credential_type}")
        
        # Validate biometric type if provided
        biometric_type = None
        if request.biometric_type:
            from .quantum_auth import BiometricType
            try:
                biometric_type = BiometricType(request.biometric_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported biometric type: {request.biometric_type}")
        
        # Register credential
        credential_id = await quantum_auth.register_user_credential(
            user_id=request.user_id,
            credential_type=credential_type,
            credential_data=request.credential_data,
            biometric_template=request.biometric_template,
            biometric_type=biometric_type
        )
        
        return {
            "credential_id": credential_id,
            "user_id": request.user_id,
            "credential_type": credential_type.value,
            "registration_time": datetime.now(timezone.utc).isoformat(),
            "status": "registered"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering credential: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to register credential")


@router.post("/auth/authenticate", summary="Authenticate user")
async def authenticate_user(request: AuthenticationRequest):
    """Authenticate a user and create quantum-safe session."""
    try:
        # Validate authentication method
        try:
            auth_method = AuthenticationMethod(request.authentication_method.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported authentication method: {request.authentication_method}")
        
        # Authenticate user
        session_id = await quantum_auth.authenticate_user(
            user_id=request.user_id,
            authentication_data=request.authentication_data,
            authentication_method=auth_method,
            client_info=request.client_info
        )
        
        if not session_id:
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get session information
        session = await quantum_auth.validate_session(session_id)
        
        return {
            "session_id": session_id,
            "user_id": request.user_id,
            "authentication_method": auth_method.value,
            "session_type": session.session_type.value,
            "authorization_level": session.authorization_level.value,
            "creation_time": session.creation_time.isoformat(),
            "expiry_time": session.expiry_time.isoformat(),
            "quantum_secured": session.quantum_key_id is not None,
            "mfa_verified": session.mfa_verified,
            "biometric_verified": session.biometric_verified
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to authenticate user")


@router.get("/auth/validate-session/{session_id}", summary="Validate session")
async def validate_session(session_id: str):
    """Validate a quantum-safe session."""
    try:
        session = await quantum_auth.validate_session(session_id)
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "valid": True,
            "session_type": session.session_type.value,
            "authorization_level": session.authorization_level.value,
            "last_activity": session.last_activity.isoformat(),
            "expires_at": session.expiry_time.isoformat(),
            "quantum_secured": session.quantum_key_id is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to validate session")


@router.post("/auth/logout/{session_id}", summary="Logout and invalidate session")
async def logout_session(session_id: str):
    """Logout and invalidate a quantum-safe session."""
    try:
        await quantum_auth.invalidate_session(session_id, "user_logout")
        
        return {
            "session_id": session_id,
            "status": "logged_out",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error logging out session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to logout session")


@router.post("/auth/authorize/{session_id}/{permission}", summary="Check authorization")
async def check_authorization(session_id: str, permission: str):
    """Check if session is authorized for a specific action."""
    try:
        is_authorized = await quantum_auth.authorize_action(session_id, permission)
        
        return {
            "session_id": session_id,
            "permission": permission,
            "authorized": is_authorized,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking authorization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check authorization")


# Quantum Threat Detection Routes

@router.get("/threats/status", summary="Get threat detection status")
async def get_threat_status():
    """Get quantum threat detection system status."""
    try:
        return quantum_threats.get_threat_status()
    except Exception as e:
        logger.error(f"Error getting threat status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get threat status")


@router.post("/threats/detect", summary="Detect quantum threats")
async def detect_quantum_threats(request: ThreatDetectionRequest):
    """Analyze system data for quantum threats."""
    try:
        threats = await quantum_threats.detect_quantum_threats(
            system_data=request.system_data,
            network_traffic=request.network_traffic,
            computational_metrics=request.computational_metrics
        )
        
        # Format threats for response
        formatted_threats = []
        for threat in threats:
            formatted_threats.append({
                "threat_id": threat.threat_id,
                "threat_type": threat.threat_type.value,
                "severity": threat.severity.value,
                "confidence": threat.confidence,
                "description": threat.description,
                "affected_algorithms": [algo.value for algo in threat.affected_algorithms],
                "affected_systems": threat.affected_systems,
                "detection_time": threat.detection_time.isoformat(),
                "source": threat.source,
                "recommended_actions": [action.value for action in threat.recommended_actions],
                "quantum_indicators": threat.quantum_indicators
            })
        
        return {
            "threats_detected": len(threats),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "threats": formatted_threats
        }
        
    except Exception as e:
        logger.error(f"Error detecting quantum threats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to detect quantum threats")


@router.post("/threats/report", summary="Generate threat report")
async def generate_threat_report(
    start_date: str,
    end_date: str,
    include_details: bool = True
):
    """Generate comprehensive quantum threat report."""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Generate report
        report = await quantum_threats.generate_threat_report(
            start_date=start_dt,
            end_date=end_dt,
            include_details=include_details
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating threat report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate threat report")


# Health check endpoint
@router.get("/health", summary="Quantum-safe system health check")
async def health_check():
    """Comprehensive health check for all quantum-safe components."""
    try:
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check each component
        try:
            pq_status = pq_crypto.get_status()
            health_status["components"]["post_quantum_crypto"] = {
                "status": "healthy" if pq_status["system_status"] == "operational" else "unhealthy",
                "details": pq_status
            }
        except Exception as e:
            health_status["components"]["post_quantum_crypto"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            qkd_status = qkd_manager.get_system_status()
            health_status["components"]["quantum_key_distribution"] = {
                "status": "healthy" if qkd_status["system_status"] == "operational" else "unhealthy",
                "details": qkd_status
            }
        except Exception as e:
            health_status["components"]["quantum_key_distribution"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            audit_status = quantum_audit.get_blockchain_status()
            health_status["components"]["quantum_audit"] = {
                "status": "healthy" if audit_status["status"] == "operational" else "unhealthy",
                "details": audit_status
            }
        except Exception as e:
            health_status["components"]["quantum_audit"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            auth_status = quantum_auth.get_authentication_status()
            health_status["components"]["quantum_auth"] = {
                "status": "healthy" if auth_status["system_status"] == "operational" else "unhealthy",
                "details": auth_status
            }
        except Exception as e:
            health_status["components"]["quantum_auth"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            threat_status = quantum_threats.get_threat_status()
            health_status["components"]["quantum_threats"] = {
                "status": "healthy" if threat_status["system_status"] == "monitoring" else "unhealthy",
                "details": threat_status
            }
        except Exception as e:
            health_status["components"]["quantum_threats"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        unhealthy_components = [
            comp for comp, details in health_status["components"].items()
            if details["status"] == "unhealthy"
        ]
        
        if unhealthy_components:
            health_status["overall_status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "error",
            "error": str(e)
        }


# Add router to main application
def setup_quantum_safe_routes(app):
    """Setup quantum-safe routes in FastAPI application."""
    app.include_router(router)
    logger.info("Quantum-safe security routes initialized")
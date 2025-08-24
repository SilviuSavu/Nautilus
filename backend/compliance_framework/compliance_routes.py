"""
Compliance Framework API Routes
===============================

FastAPI routes for the advanced compliance and regulatory framework,
providing comprehensive API access to all compliance features.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import json
import io

from .policy_engine import (
    CompliancePolicyEngine, CompliancePolicy, JurisdictionType, 
    ComplianceFramework, DataClassification, RiskLevel
)
from .audit_trail import (
    ImmutableAuditTrail, AuditEvent, EventType, SeverityLevel as AuditSeverity, DataCategory
)
from .reporting_engine import (
    AutomatedComplianceReporter, ReportConfiguration, ReportType, 
    ReportFormat, ReportFrequency
)
from .data_residency import (
    DataResidencyController, DataRecord, Jurisdiction, DataLocation, 
    EncryptionLevel, TransferMechanism
)
from .monitoring_system import (
    RealTimeComplianceMonitor, MonitoringRule, ComplianceViolation, 
    ViolationType, SeverityLevel, MonitoringStatus
)

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])

# Global instances - in production these would be dependency injected
policy_engine = CompliancePolicyEngine()
audit_trail = ImmutableAuditTrail()
reporting_engine = AutomatedComplianceReporter()
residency_controller = DataResidencyController()
monitoring_system = RealTimeComplianceMonitor()


# Pydantic Models for API
class PolicyEvaluationRequest(BaseModel):
    event_data: Dict[str, Any]
    jurisdiction: str
    data_classification: str


class PolicyEvaluationResponse(BaseModel):
    evaluation_id: str
    timestamp: str
    jurisdiction: str
    data_classification: str
    compliance_score: float
    is_compliant: bool
    violations: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    recommendations: List[str]


class AuditEventRequest(BaseModel):
    event_type: str
    severity: str
    resource_accessed: str
    action_performed: str
    data_category: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_details: Optional[Dict[str, Any]] = None
    response_details: Optional[Dict[str, Any]] = None
    compliance_context: Optional[Dict[str, Any]] = None
    additional_fields: Optional[Dict[str, Any]] = None


class ReportGenerationRequest(BaseModel):
    report_id: str
    custom_start_date: Optional[str] = None
    custom_end_date: Optional[str] = None


class DataClassificationRequest(BaseModel):
    data_content: str
    data_category: str
    subject_jurisdiction: str
    business_purpose: str
    legal_basis: str
    data_subject_id: Optional[str] = None


class TransferValidationRequest(BaseModel):
    record_id: str
    destination_jurisdiction: str
    transfer_purpose: str


class TransferApprovalRequest(BaseModel):
    record_id: str
    destination_jurisdiction: str
    transfer_mechanism: str
    business_justification: str
    requested_by: str


class MonitoringRuleRequest(BaseModel):
    name: str
    description: str
    violation_type: str
    severity: str
    condition: str
    data_sources: List[str]
    monitoring_frequency_seconds: int
    threshold_values: Dict[str, Any]
    evaluation_window_minutes: int
    alert_channels: List[str]
    auto_remediation_enabled: bool = False


# ==================== POLICY ENGINE ROUTES ====================

@router.get("/policy/status")
async def get_policy_engine_status():
    """Get policy engine status and statistics"""
    status = policy_engine.get_compliance_status()
    return JSONResponse(content=status)


@router.post("/policy/evaluate", response_model=PolicyEvaluationResponse)
async def evaluate_compliance_policy(request: PolicyEvaluationRequest):
    """Evaluate compliance for an event against applicable policies"""
    
    try:
        jurisdiction = JurisdictionType(request.jurisdiction)
        data_classification = DataClassification(request.data_classification)
        
        result = policy_engine.evaluate_compliance(
            request.event_data,
            jurisdiction,
            data_classification
        )
        
        return PolicyEvaluationResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy evaluation failed: {str(e)}")


@router.get("/policy/applicable")
async def get_applicable_policies(
    jurisdiction: Optional[str] = Query(None),
    framework: Optional[str] = Query(None),
    data_classification: Optional[str] = Query(None)
):
    """Get applicable policies based on criteria"""
    
    try:
        jurisdiction_enum = JurisdictionType(jurisdiction) if jurisdiction else None
        framework_enum = ComplianceFramework(framework) if framework else None
        classification_enum = DataClassification(data_classification) if data_classification else None
        
        policies = policy_engine.get_applicable_policies(
            jurisdiction=jurisdiction_enum,
            framework=framework_enum,
            data_classification=classification_enum
        )
        
        # Convert to serializable format
        result = []
        for policy in policies:
            policy_dict = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "jurisdiction": policy.jurisdiction.value,
                "framework": policy.framework.value,
                "description": policy.description,
                "risk_level": policy.risk_level.value,
                "data_classification": policy.data_classification.value,
                "automated_enforcement": policy.automated_enforcement,
                "effective_date": policy.effective_date.isoformat(),
                "expiry_date": policy.expiry_date.isoformat() if policy.expiry_date else None
            }
            result.append(policy_dict)
        
        return JSONResponse(content={"policies": result, "count": len(result)})
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve policies: {str(e)}")


# ==================== AUDIT TRAIL ROUTES ====================

@router.post("/audit/log")
async def log_audit_event(request: AuditEventRequest):
    """Log an audit event with immutable characteristics"""
    
    try:
        event_type = EventType(request.event_type)
        severity = AuditSeverity(request.severity)
        data_category = DataCategory(request.data_category)
        
        event_id = audit_trail.log_event(
            event_type=event_type,
            severity=severity,
            resource_accessed=request.resource_accessed,
            action_performed=request.action_performed,
            data_category=data_category,
            user_id=request.user_id,
            session_id=request.session_id,
            source_ip=request.source_ip,
            user_agent=request.user_agent,
            request_details=request.request_details,
            response_details=request.response_details,
            compliance_context=request.compliance_context,
            **(request.additional_fields or {})
        )
        
        return JSONResponse(content={
            "event_id": event_id,
            "status": "logged",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit logging failed: {str(e)}")


@router.get("/audit/verify")
async def verify_audit_chain_integrity():
    """Verify the integrity of the entire audit chain"""
    
    try:
        verification_result = audit_trail.verify_chain_integrity()
        return JSONResponse(content=verification_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain verification failed: {str(e)}")


@router.get("/audit/query")
async def query_audit_events(
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    data_category: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=10000)
):
    """Query audit events with filters"""
    
    try:
        # Parse optional filters
        event_type_enum = EventType(event_type) if event_type else None
        severity_enum = AuditSeverity(severity) if severity else None
        data_category_enum = DataCategory(data_category) if data_category else None
        start_datetime = datetime.fromisoformat(start_time) if start_time else None
        end_datetime = datetime.fromisoformat(end_time) if end_time else None
        
        events = audit_trail.query_events(
            event_type=event_type_enum,
            severity=severity_enum,
            user_id=user_id,
            start_time=start_datetime,
            end_time=end_datetime,
            data_category=data_category_enum,
            limit=limit
        )
        
        # Convert events to serializable format
        result = []
        for event in events:
            event_dict = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "user_id": event.user_id,
                "resource_accessed": event.resource_accessed,
                "action_performed": event.action_performed,
                "data_category": event.data_category.value,
                "jurisdiction": event.jurisdiction,
                "current_hash": event.current_hash,
                "verification_status": event.verification_status
            }
            result.append(event_dict)
        
        return JSONResponse(content={"events": result, "count": len(result)})
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/audit/metrics")
async def get_audit_metrics():
    """Get audit system performance metrics"""
    
    try:
        metrics = audit_trail.get_audit_metrics()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/audit/compliance-report/{jurisdiction}")
async def get_audit_compliance_report(
    jurisdiction: str,
    start_date: str,
    end_date: str
):
    """Generate compliance report for specific jurisdiction"""
    
    try:
        start_datetime = datetime.fromisoformat(start_date)
        end_datetime = datetime.fromisoformat(end_date)
        
        report = audit_trail.get_compliance_report(
            jurisdiction=jurisdiction,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        return JSONResponse(content=report)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ==================== REPORTING ENGINE ROUTES ====================

@router.post("/reports/generate")
async def generate_compliance_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a compliance report"""
    
    try:
        custom_period = None
        if request.custom_start_date and request.custom_end_date:
            start_date = datetime.fromisoformat(request.custom_start_date)
            end_date = datetime.fromisoformat(request.custom_end_date)
            custom_period = (start_date, end_date)
        
        # Generate report in background
        metadata = await reporting_engine.generate_report(
            request.report_id,
            custom_period
        )
        
        if metadata:
            return JSONResponse(content={
                "generation_id": metadata.generation_id,
                "report_id": metadata.report_id,
                "status": "generated",
                "file_path": metadata.file_path,
                "generation_time_seconds": metadata.generation_time_seconds,
                "compliance_status": metadata.compliance_status
            })
        else:
            raise HTTPException(status_code=500, detail="Report generation failed")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/reports/configurations")
async def list_report_configurations():
    """List all active report configurations"""
    
    try:
        configurations = reporting_engine.list_active_configurations()
        return JSONResponse(content={"configurations": configurations})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list configurations: {str(e)}")


@router.get("/reports/status/{generation_id}")
async def get_report_status(generation_id: str):
    """Get status of a specific report generation"""
    
    try:
        status = reporting_engine.get_report_status(generation_id)
        
        if status:
            return JSONResponse(content=status)
        else:
            raise HTTPException(status_code=404, detail="Report generation not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report status: {str(e)}")


# ==================== DATA RESIDENCY ROUTES ====================

@router.post("/data/classify")
async def classify_data_record(request: DataClassificationRequest):
    """Classify data and determine residency requirements"""
    
    try:
        data_category = DataCategory(request.data_category)
        jurisdiction = Jurisdiction(request.subject_jurisdiction)
        
        record = residency_controller.classify_data(
            data_content=request.data_content,
            data_category=data_category,
            subject_jurisdiction=jurisdiction,
            business_purpose=request.business_purpose,
            legal_basis=request.legal_basis,
            data_subject_id=request.data_subject_id
        )
        
        # Convert to serializable format
        result = {
            "record_id": record.record_id,
            "data_category": record.data_category.value,
            "classification": record.classification.value,
            "subject_jurisdiction": record.subject_jurisdiction.value,
            "data_location": record.data_location.value,
            "encryption_level": record.encryption_level.value,
            "retention_period_days": record.retention_period_days,
            "deletion_date": record.deletion_date.isoformat() if record.deletion_date else None,
            "pseudonymized": record.pseudonymized,
            "anonymized": record.anonymized
        }
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data classification failed: {str(e)}")


@router.post("/data/transfer/validate")
async def validate_data_transfer(request: TransferValidationRequest):
    """Validate if a cross-border data transfer is compliant"""
    
    try:
        destination_jurisdiction = Jurisdiction(request.destination_jurisdiction)
        
        validation_result = residency_controller.validate_transfer(
            record_id=request.record_id,
            destination_jurisdiction=destination_jurisdiction,
            transfer_purpose=request.transfer_purpose
        )
        
        return JSONResponse(content=validation_result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid jurisdiction: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer validation failed: {str(e)}")


@router.post("/data/transfer/approve")
async def request_transfer_approval(request: TransferApprovalRequest):
    """Request approval for cross-border data transfer"""
    
    try:
        destination_jurisdiction = Jurisdiction(request.destination_jurisdiction)
        transfer_mechanism = TransferMechanism(request.transfer_mechanism)
        
        approval_id = residency_controller.request_transfer_approval(
            record_id=request.record_id,
            destination_jurisdiction=destination_jurisdiction,
            transfer_mechanism=transfer_mechanism,
            business_justification=request.business_justification,
            requested_by=request.requested_by
        )
        
        if approval_id:
            return JSONResponse(content={
                "approval_id": approval_id,
                "status": "approved",
                "valid_until": (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail="Transfer approval denied")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer approval failed: {str(e)}")


@router.get("/data/compliance/status")
async def get_data_compliance_status(
    jurisdiction: Optional[str] = Query(None),
    data_category: Optional[str] = Query(None)
):
    """Get data compliance status overview"""
    
    try:
        jurisdiction_enum = Jurisdiction(jurisdiction) if jurisdiction else None
        data_category_enum = DataCategory(data_category) if data_category else None
        
        status = residency_controller.get_compliance_status(
            jurisdiction=jurisdiction_enum,
            data_category=data_category_enum
        )
        
        return JSONResponse(content=status)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compliance status: {str(e)}")


@router.get("/data/compliance/report/{jurisdiction}")
async def export_data_compliance_report(
    jurisdiction: str,
    start_date: str,
    end_date: str
):
    """Export comprehensive data compliance report"""
    
    try:
        jurisdiction_enum = Jurisdiction(jurisdiction)
        start_datetime = datetime.fromisoformat(start_date)
        end_datetime = datetime.fromisoformat(end_date)
        
        report = residency_controller.export_compliance_report(
            jurisdiction=jurisdiction_enum,
            start_date=start_datetime,
            end_date=end_datetime
        )
        
        return JSONResponse(content=report)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")


# ==================== MONITORING SYSTEM ROUTES ====================

@router.get("/monitoring/status")
async def get_monitoring_system_status():
    """Get real-time monitoring system status"""
    
    try:
        status = monitoring_system.get_system_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/monitoring/violations")
async def get_violations_summary():
    """Get summary of compliance violations"""
    
    try:
        summary = monitoring_system.get_violations_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violations summary: {str(e)}")


@router.get("/monitoring/violations/active")
async def get_active_violations(
    severity: Optional[str] = Query(None),
    violation_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get active compliance violations"""
    
    try:
        violations = list(monitoring_system.active_violations.values())
        
        # Apply filters
        if severity:
            severity_enum = SeverityLevel(severity)
            violations = [v for v in violations if v.severity == severity_enum]
        
        if violation_type:
            violation_type_enum = ViolationType(violation_type)
            violations = [v for v in violations if v.violation_type == violation_type_enum]
        
        # Limit results
        violations = violations[:limit]
        
        # Convert to serializable format
        result = []
        for violation in violations:
            violation_dict = {
                "violation_id": violation.violation_id,
                "violation_type": violation.violation_type.value,
                "severity": violation.severity.value,
                "title": violation.title,
                "description": violation.description,
                "affected_system": violation.affected_system,
                "jurisdiction": violation.jurisdiction,
                "regulatory_framework": violation.regulatory_framework,
                "detection_timestamp": violation.detection_timestamp.isoformat(),
                "status": violation.status,
                "resolution_due": violation.resolution_due.isoformat(),
                "risk_score": violation.risk_score,
                "business_impact": violation.business_impact,
                "auto_remediation_attempted": violation.auto_remediation_attempted
            }
            result.append(violation_dict)
        
        return JSONResponse(content={"violations": result, "count": len(result)})
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get violations: {str(e)}")


@router.post("/monitoring/rules")
async def add_monitoring_rule(request: MonitoringRuleRequest):
    """Add a new monitoring rule"""
    
    try:
        from .monitoring_system import AlertChannel
        
        violation_type = ViolationType(request.violation_type)
        severity = SeverityLevel(request.severity)
        alert_channels = [AlertChannel(ch) for ch in request.alert_channels]
        
        rule = MonitoringRule(
            rule_id=f"CUSTOM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            name=request.name,
            description=request.description,
            violation_type=violation_type,
            severity=severity,
            condition=request.condition,
            data_sources=request.data_sources,
            monitoring_frequency_seconds=request.monitoring_frequency_seconds,
            threshold_values=request.threshold_values,
            evaluation_window_minutes=request.evaluation_window_minutes,
            aggregation_method="count",
            baseline_values={},
            alert_channels=alert_channels,
            auto_remediation_enabled=request.auto_remediation_enabled,
            escalation_rules=[],
            business_hours_only=False,
            maintenance_windows=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_evaluation=None,
            evaluation_count=0,
            violation_count=0,
            false_positive_count=0,
            is_active=True
        )
        
        success = monitoring_system.add_monitoring_rule(rule)
        
        if success:
            return JSONResponse(content={
                "rule_id": rule.rule_id,
                "status": "added",
                "name": rule.name
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to add monitoring rule")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add monitoring rule: {str(e)}")


@router.get("/monitoring/rules")
async def get_monitoring_rules():
    """Get all monitoring rules"""
    
    try:
        rules = []
        for rule in monitoring_system.monitoring_rules.values():
            rule_dict = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "violation_type": rule.violation_type.value,
                "severity": rule.severity.value,
                "monitoring_frequency_seconds": rule.monitoring_frequency_seconds,
                "is_active": rule.is_active,
                "evaluation_count": rule.evaluation_count,
                "violation_count": rule.violation_count,
                "last_evaluation": rule.last_evaluation.isoformat() if rule.last_evaluation else None,
                "auto_remediation_enabled": rule.auto_remediation_enabled
            }
            rules.append(rule_dict)
        
        return JSONResponse(content={"rules": rules, "count": len(rules)})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring rules: {str(e)}")


# ==================== SYSTEM-WIDE ROUTES ====================

@router.get("/health")
async def compliance_system_health():
    """Get overall compliance system health"""
    
    try:
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "components": {
                "policy_engine": {
                    "status": "healthy",
                    "active_policies": len(policy_engine.policies),
                    "active_violations": len(policy_engine.active_violations)
                },
                "audit_trail": {
                    "status": "healthy",
                    "total_events": audit_trail.metrics["total_events"],
                    "integrity_status": "verified" if audit_trail.verify_chain_integrity()["is_valid"] else "compromised"
                },
                "reporting_engine": {
                    "status": "healthy",
                    "active_configurations": len([c for c in reporting_engine.configurations.values() if c.is_active])
                },
                "data_residency": {
                    "status": "healthy",
                    "managed_records": len(residency_controller.data_records),
                    "active_rules": len([r for r in residency_controller.residency_rules.values() if r.is_active])
                },
                "monitoring_system": {
                    "status": monitoring_system.status.value,
                    "active_rules": len([r for r in monitoring_system.monitoring_rules.values() if r.is_active]),
                    "open_violations": len(monitoring_system.active_violations)
                }
            }
        }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "error" in component_statuses or "compromised" in [
            health_status["components"]["audit_trail"]["integrity_status"]
        ]:
            health_status["overall_status"] = "degraded"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.get("/metrics")
async def get_compliance_metrics():
    """Get comprehensive compliance metrics"""
    
    try:
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_engine": policy_engine.get_compliance_status(),
            "audit_trail": audit_trail.get_audit_metrics(),
            "data_residency": residency_controller.get_compliance_status(),
            "monitoring_system": monitoring_system.get_system_status(),
            "violations_summary": monitoring_system.get_violations_summary()
        }
        
        return JSONResponse(content=metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compliance metrics: {str(e)}")


@router.get("/dashboard")
async def get_compliance_dashboard():
    """Get compliance dashboard data"""
    
    try:
        # Aggregate data from all components
        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_policies": len(policy_engine.policies),
                "total_audit_events": audit_trail.metrics.get("total_events", 0),
                "total_data_records": len(residency_controller.data_records),
                "active_violations": len(monitoring_system.active_violations),
                "critical_violations": len([
                    v for v in monitoring_system.active_violations.values() 
                    if v.severity == SeverityLevel.CRITICAL
                ]),
                "compliance_score": max(0, 100 - (len(monitoring_system.active_violations) * 5))
            },
            "recent_activity": {
                "violations_last_24h": len([
                    v for v in monitoring_system.violation_history
                    if (datetime.now(timezone.utc) - v.detection_timestamp).total_seconds() < 86400
                ]),
                "audit_events_last_1h": len([
                    e for block in audit_trail.blocks for e in block.events
                    if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600
                ]) + len([
                    e for e in audit_trail.pending_events
                    if (datetime.now(timezone.utc) - e.timestamp).total_seconds() < 3600
                ]),
                "data_classifications_today": len([
                    r for r in residency_controller.data_records.values()
                    if (datetime.now(timezone.utc) - r.created_at).total_seconds() < 86400
                ])
            },
            "top_violations": [
                {
                    "violation_id": v.violation_id,
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "title": v.title,
                    "risk_score": v.risk_score,
                    "detection_time": v.detection_timestamp.isoformat()
                }
                for v in sorted(
                    monitoring_system.active_violations.values(),
                    key=lambda x: (x.severity == SeverityLevel.CRITICAL, x.risk_score),
                    reverse=True
                )[:5]
            ]
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/export/comprehensive-report")
async def export_comprehensive_compliance_report(
    jurisdiction: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    format: str = Query("json", regex="^(json|csv|pdf)$")
):
    """Export comprehensive compliance report"""
    
    try:
        # Default date range - last 30 days
        end_datetime = datetime.now(timezone.utc)
        start_datetime = end_datetime - timedelta(days=30)
        
        if start_date:
            start_datetime = datetime.fromisoformat(start_date)
        if end_date:
            end_datetime = datetime.fromisoformat(end_date)
        
        # Collect data from all systems
        comprehensive_report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_period": {
                    "start_date": start_datetime.isoformat(),
                    "end_date": end_datetime.isoformat()
                },
                "jurisdiction_filter": jurisdiction,
                "format": format
            },
            "executive_summary": {
                "total_policies": len(policy_engine.policies),
                "total_violations": len(monitoring_system.active_violations),
                "critical_violations": len([
                    v for v in monitoring_system.active_violations.values()
                    if v.severity == SeverityLevel.CRITICAL
                ]),
                "compliance_score": max(0, 100 - (len(monitoring_system.active_violations) * 5)),
                "audit_integrity_status": "verified" if audit_trail.verify_chain_integrity()["is_valid"] else "compromised"
            },
            "policy_compliance": policy_engine.get_compliance_status(),
            "audit_summary": audit_trail.get_compliance_report(
                jurisdiction or "GLOBAL",
                start_datetime,
                end_datetime
            ),
            "data_residency": residency_controller.export_compliance_report(
                Jurisdiction(jurisdiction) if jurisdiction else Jurisdiction.GLOBAL,
                start_datetime,
                end_datetime
            ),
            "monitoring_status": monitoring_system.get_system_status(),
            "violations_analysis": monitoring_system.get_violations_summary()
        }
        
        if format == "json":
            return JSONResponse(content=comprehensive_report)
        elif format == "csv":
            # Convert to CSV format
            csv_output = io.StringIO()
            # Simplified CSV export - in production would use proper CSV formatting
            csv_output.write("Component,Metric,Value\n")
            csv_output.write(f"Executive,Total Policies,{comprehensive_report['executive_summary']['total_policies']}\n")
            csv_output.write(f"Executive,Total Violations,{comprehensive_report['executive_summary']['total_violations']}\n")
            csv_output.write(f"Executive,Compliance Score,{comprehensive_report['executive_summary']['compliance_score']}\n")
            
            return StreamingResponse(
                io.BytesIO(csv_output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=compliance_report.csv"}
            )
        else:
            # PDF format would require additional libraries
            raise HTTPException(status_code=501, detail="PDF format not yet implemented")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report export failed: {str(e)}")
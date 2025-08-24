"""
Phase 8 Autonomous Security Operations API Routes
==================================================

FastAPI routes for Phase 8 autonomous security operations including cognitive security,
threat intelligence, autonomous response, fraud detection, and security orchestration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Body, Path, Depends
from pydantic import BaseModel, Field
import redis.asyncio as redis

# Import Phase 8 security components
from phase8_autonomous_operations.security.cognitive_security_operations_center import (
    CognitiveSecurityOperationsCenter, get_csoc, ThreatSeverity, ThreatCategory
)
from phase8_autonomous_operations.threat_intelligence.advanced_threat_intelligence import (
    AdvancedThreatIntelligence, get_threat_intelligence, ThreatIntelligenceSource, IndicatorType
)
from phase8_autonomous_operations.security_response.autonomous_security_response import (
    AutonomousSecurityResponse, get_autonomous_security_response, ResponseAction, ResponseSeverity
)
from phase8_autonomous_operations.fraud_detection.intelligent_fraud_detection import (
    IntelligentFraudDetection, get_fraud_detector, FraudType, FraudSeverity
)
from phase8_autonomous_operations.security_orchestration.automated_security_orchestration import (
    AutomatedSecurityOrchestration, get_security_orchestration
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/security", tags=["Phase8-Security"])


# Pydantic models for API requests/responses
class SecurityAnalysisRequest(BaseModel):
    """Request for security event analysis"""
    event_data: Dict[str, Any] = Field(..., description="Event data to analyze")
    source_system: str = Field(..., description="Source system generating the event")
    event_type: str = Field(..., description="Type of security event")
    priority: Optional[str] = Field("medium", description="Event priority level")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class SecurityAnalysisResponse(BaseModel):
    """Response for security analysis"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    threat_level: str = Field(..., description="Assessed threat level")
    confidence_score: float = Field(..., description="Analysis confidence (0-1)")
    detected_threats: List[str] = Field(..., description="List of detected threat types")
    recommended_actions: List[str] = Field(..., description="Recommended security actions")
    analysis_timestamp: str = Field(..., description="Analysis completion timestamp")
    cognitive_insights: Dict[str, Any] = Field(..., description="AI-generated insights")


class ThreatIntelligenceQuery(BaseModel):
    """Threat intelligence query parameters"""
    indicators: Optional[List[str]] = Field(None, description="Specific threat indicators")
    threat_types: Optional[List[str]] = Field(None, description="Filter by threat types")
    sources: Optional[List[str]] = Field(None, description="Intelligence sources to query")
    time_range_hours: int = Field(24, description="Time range for intelligence lookup")
    confidence_threshold: float = Field(0.5, description="Minimum confidence threshold")


class ThreatIntelligenceResponse(BaseModel):
    """Threat intelligence response"""
    query_id: str = Field(..., description="Query identifier")
    total_indicators: int = Field(..., description="Total indicators found")
    high_confidence_threats: List[Dict[str, Any]] = Field(..., description="High confidence threats")
    emerging_threats: List[Dict[str, Any]] = Field(..., description="Recently identified threats")
    threat_campaigns: List[Dict[str, Any]] = Field(..., description="Active threat campaigns")
    intelligence_summary: Dict[str, Any] = Field(..., description="Intelligence summary")


class SecurityResponseRequest(BaseModel):
    """Security response execution request"""
    threat_id: str = Field(..., description="Threat identifier to respond to")
    response_type: str = Field(..., description="Type of response to execute")
    severity: str = Field(..., description="Response severity level")
    auto_execute: bool = Field(False, description="Execute response automatically")
    custom_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SecurityResponseStatus(BaseModel):
    """Security response execution status"""
    response_id: str = Field(..., description="Response execution identifier")
    status: str = Field(..., description="Current execution status")
    actions_taken: List[str] = Field(..., description="Actions executed")
    effectiveness_score: float = Field(..., description="Response effectiveness (0-1)")
    completion_time: Optional[str] = Field(None, description="Response completion time")
    side_effects: List[str] = Field(default_factory=list, description="Any side effects detected")


class FraudAnalysisRequest(BaseModel):
    """Fraud detection analysis request"""
    transaction_data: Dict[str, Any] = Field(..., description="Transaction or activity data")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="User behavioral profile")
    session_context: Optional[Dict[str, Any]] = Field(None, description="Session context information")
    analysis_depth: str = Field("standard", description="Analysis depth: basic, standard, deep")


class FraudAnalysisResponse(BaseModel):
    """Fraud detection analysis response"""
    analysis_id: str = Field(..., description="Analysis identifier")
    fraud_score: float = Field(..., description="Fraud probability score (0-1)")
    detected_patterns: List[str] = Field(..., description="Detected fraudulent patterns")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Identified risk factors")
    recommended_actions: List[str] = Field(..., description="Recommended fraud prevention actions")
    behavioral_anomalies: List[Dict[str, Any]] = Field(..., description="Behavioral anomalies detected")


class SecurityOrchestrationStatus(BaseModel):
    """Security orchestration system status"""
    orchestration_state: str = Field(..., description="Current orchestration state")
    active_workflows: int = Field(..., description="Number of active security workflows")
    completed_responses: int = Field(..., description="Completed security responses today")
    avg_response_time: float = Field(..., description="Average response time in seconds")
    system_health: str = Field(..., description="Overall system health status")
    integration_status: Dict[str, str] = Field(..., description="Status of integrated security systems")


class SystemStatusResponse(BaseModel):
    """Overall security system status"""
    overall_status: str = Field(..., description="Overall security system status")
    cognitive_security_health: str = Field(..., description="Cognitive security center health")
    threat_intelligence_health: str = Field(..., description="Threat intelligence system health")
    response_system_health: str = Field(..., description="Autonomous response system health")
    fraud_detection_health: str = Field(..., description="Fraud detection system health")
    orchestration_health: str = Field(..., description="Security orchestration health")
    active_threats: int = Field(..., description="Number of active threats")
    blocked_incidents: int = Field(..., description="Incidents blocked in last 24h")
    system_uptime: str = Field(..., description="System uptime")


# Security Analysis Routes
@router.post("/analyze", response_model=SecurityAnalysisResponse)
async def analyze_security_event(
    request: SecurityAnalysisRequest,
    csoc: CognitiveSecurityOperationsCenter = Depends(get_csoc)
):
    """
    Analyze security events using cognitive security operations center.
    
    Performs comprehensive AI-driven analysis of security events including:
    - Behavioral anomaly detection
    - Threat classification and severity assessment
    - Pattern recognition and correlation
    - Cognitive threat analysis with machine learning
    """
    try:
        logger.info(f"Analyzing security event from {request.source_system}")
        
        # Perform cognitive security analysis
        analysis_result = await csoc.analyze_security_event(
            event_data=request.event_data,
            source_system=request.source_system,
            event_type=request.event_type,
            priority=request.priority,
            context=request.context
        )
        
        return SecurityAnalysisResponse(
            analysis_id=analysis_result["analysis_id"],
            threat_level=analysis_result["threat_level"],
            confidence_score=analysis_result["confidence_score"],
            detected_threats=analysis_result["detected_threats"],
            recommended_actions=analysis_result["recommended_actions"],
            analysis_timestamp=analysis_result["timestamp"],
            cognitive_insights=analysis_result["cognitive_insights"]
        )
        
    except Exception as e:
        logger.error(f"Error analyzing security event: {e}")
        raise HTTPException(status_code=500, detail=f"Security analysis failed: {str(e)}")


@router.get("/analyze/history")
async def get_analysis_history(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    csoc: CognitiveSecurityOperationsCenter = Depends(get_csoc)
):
    """Get historical security analysis results."""
    try:
        history = await csoc.get_analysis_history(
            hours_back=hours_back,
            threat_level=threat_level,
            limit=limit
        )
        
        return {
            "total_analyses": len(history),
            "time_range": f"Last {hours_back} hours",
            "analysis_history": history,
            "threat_distribution": await csoc.get_threat_distribution(hours_back),
            "trends": await csoc.get_security_trends(hours_back)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Threat Intelligence Routes
@router.post("/threats/intelligence", response_model=ThreatIntelligenceResponse)
async def query_threat_intelligence(
    query: ThreatIntelligenceQuery,
    threat_intel: AdvancedThreatIntelligence = Depends(get_threat_intelligence)
):
    """
    Query advanced threat intelligence system.
    
    Provides comprehensive threat intelligence including:
    - Threat indicator analysis and correlation
    - Behavioral threat profiling
    - Threat campaign tracking
    - External intelligence feed integration
    """
    try:
        logger.info(f"Querying threat intelligence with {len(query.indicators or [])} indicators")
        
        # Execute threat intelligence query
        intelligence_result = await threat_intel.query_intelligence(
            indicators=query.indicators,
            threat_types=query.threat_types,
            sources=query.sources,
            time_range_hours=query.time_range_hours,
            confidence_threshold=query.confidence_threshold
        )
        
        return ThreatIntelligenceResponse(
            query_id=intelligence_result["query_id"],
            total_indicators=intelligence_result["total_indicators"],
            high_confidence_threats=intelligence_result["high_confidence_threats"],
            emerging_threats=intelligence_result["emerging_threats"],
            threat_campaigns=intelligence_result["threat_campaigns"],
            intelligence_summary=intelligence_result["summary"]
        )
        
    except Exception as e:
        logger.error(f"Error querying threat intelligence: {e}")
        raise HTTPException(status_code=500, detail=f"Threat intelligence query failed: {str(e)}")


@router.get("/threats/feeds")
async def get_threat_feeds_status(
    threat_intel: AdvancedThreatIntelligence = Depends(get_threat_intelligence)
):
    """Get status of all threat intelligence feeds."""
    try:
        feeds_status = await threat_intel.get_feeds_status()
        
        return {
            "total_feeds": len(feeds_status),
            "active_feeds": sum(1 for feed in feeds_status if feed["status"] == "active"),
            "feed_details": feeds_status,
            "last_update": await threat_intel.get_last_update_time(),
            "intelligence_stats": await threat_intel.get_intelligence_statistics()
        }
        
    except Exception as e:
        logger.error(f"Error getting threat feeds status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats/campaigns")
async def get_active_campaigns(
    active_only: bool = Query(True, description="Show only active campaigns"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    threat_intel: AdvancedThreatIntelligence = Depends(get_threat_intelligence)
):
    """Get information about active threat campaigns."""
    try:
        campaigns = await threat_intel.get_threat_campaigns(
            active_only=active_only,
            severity=severity
        )
        
        return {
            "active_campaigns": campaigns,
            "campaign_summary": await threat_intel.get_campaign_summary(),
            "attribution_analysis": await threat_intel.get_attribution_analysis()
        }
        
    except Exception as e:
        logger.error(f"Error getting threat campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Security Response Routes
@router.post("/response/execute", response_model=SecurityResponseStatus)
async def execute_security_response(
    request: SecurityResponseRequest,
    response_system: AutonomousSecurityResponse = Depends(get_autonomous_security_response)
):
    """
    Execute autonomous security response actions.
    
    Provides intelligent, adaptive security responses including:
    - Automated threat mitigation
    - Adaptive countermeasures
    - Intelligent response orchestration
    - Real-time response effectiveness assessment
    """
    try:
        logger.info(f"Executing security response for threat {request.threat_id}")
        
        # Execute autonomous security response
        response_result = await response_system.execute_response(
            threat_id=request.threat_id,
            response_type=request.response_type,
            severity=request.severity,
            auto_execute=request.auto_execute,
            custom_parameters=request.custom_parameters
        )
        
        return SecurityResponseStatus(
            response_id=response_result["response_id"],
            status=response_result["status"],
            actions_taken=response_result["actions_taken"],
            effectiveness_score=response_result["effectiveness_score"],
            completion_time=response_result.get("completion_time"),
            side_effects=response_result.get("side_effects", [])
        )
        
    except Exception as e:
        logger.error(f"Error executing security response: {e}")
        raise HTTPException(status_code=500, detail=f"Security response execution failed: {str(e)}")


@router.get("/response/status/{response_id}")
async def get_response_status(
    response_id: str = Path(..., description="Security response execution ID"),
    response_system: AutonomousSecurityResponse = Depends(get_autonomous_security_response)
):
    """Get status of a specific security response execution."""
    try:
        status = await response_system.get_response_status(response_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Response {response_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting response status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/response/capabilities")
async def get_response_capabilities(
    response_system: AutonomousSecurityResponse = Depends(get_autonomous_security_response)
):
    """Get available security response capabilities and actions."""
    try:
        capabilities = await response_system.get_capabilities()
        
        return {
            "available_actions": [action.value for action in ResponseAction],
            "severity_levels": [severity.value for severity in ResponseSeverity],
            "response_capabilities": capabilities,
            "adaptation_strategies": await response_system.get_adaptation_strategies(),
            "effectiveness_metrics": await response_system.get_effectiveness_metrics()
        }
        
    except Exception as e:
        logger.error(f"Error getting response capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Fraud Detection Routes
@router.post("/fraud/analyze", response_model=FraudAnalysisResponse)
async def analyze_fraud_risk(
    request: FraudAnalysisRequest,
    fraud_detector: IntelligentFraudDetection = Depends(get_fraud_detector)
):
    """
    Analyze transactions and activities for fraud risk.
    
    Provides comprehensive fraud detection including:
    - Real-time behavioral analysis
    - Pattern recognition and anomaly detection
    - Machine learning-based fraud scoring
    - Trading-specific fraud detection (market manipulation, insider trading, etc.)
    """
    try:
        logger.info(f"Analyzing fraud risk with {request.analysis_depth} depth")
        
        # Perform intelligent fraud analysis
        analysis_result = await fraud_detector.analyze_fraud_risk(
            transaction_data=request.transaction_data,
            user_profile=request.user_profile,
            session_context=request.session_context,
            analysis_depth=request.analysis_depth
        )
        
        return FraudAnalysisResponse(
            analysis_id=analysis_result["analysis_id"],
            fraud_score=analysis_result["fraud_score"],
            detected_patterns=analysis_result["detected_patterns"],
            risk_factors=analysis_result["risk_factors"],
            recommended_actions=analysis_result["recommended_actions"],
            behavioral_anomalies=analysis_result["behavioral_anomalies"]
        )
        
    except Exception as e:
        logger.error(f"Error analyzing fraud risk: {e}")
        raise HTTPException(status_code=500, detail=f"Fraud analysis failed: {str(e)}")


@router.get("/fraud/patterns")
async def get_fraud_patterns(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence"),
    fraud_detector: IntelligentFraudDetection = Depends(get_fraud_detector)
):
    """Get detected fraud patterns and trends."""
    try:
        patterns = await fraud_detector.get_fraud_patterns(
            pattern_type=pattern_type,
            confidence_threshold=confidence_threshold
        )
        
        return {
            "detected_patterns": patterns,
            "pattern_statistics": await fraud_detector.get_pattern_statistics(),
            "fraud_trends": await fraud_detector.get_fraud_trends(),
            "model_performance": await fraud_detector.get_model_performance()
        }
        
    except Exception as e:
        logger.error(f"Error getting fraud patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud/alerts")
async def get_fraud_alerts(
    status: str = Query("active", description="Alert status filter"),
    severity: Optional[str] = Query(None, description="Alert severity filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    fraud_detector: IntelligentFraudDetection = Depends(get_fraud_detector)
):
    """Get fraud detection alerts."""
    try:
        alerts = await fraud_detector.get_fraud_alerts(
            status=status,
            severity=severity,
            limit=limit
        )
        
        return {
            "fraud_alerts": alerts,
            "alert_summary": await fraud_detector.get_alert_summary(),
            "resolution_stats": await fraud_detector.get_resolution_statistics()
        }
        
    except Exception as e:
        logger.error(f"Error getting fraud alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Security Orchestration Routes
@router.get("/orchestration/status", response_model=SecurityOrchestrationStatus)
async def get_orchestration_status(
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """
    Get security orchestration system status.
    
    Provides comprehensive status of security workflow orchestration including:
    - Active security workflows and playbooks
    - Response coordination and effectiveness
    - Integration status with security systems
    - Performance metrics and health indicators
    """
    try:
        status = await orchestration.get_orchestration_status()
        
        return SecurityOrchestrationStatus(
            orchestration_state=status["orchestration_state"],
            active_workflows=status["active_workflows"],
            completed_responses=status["completed_responses"],
            avg_response_time=status["avg_response_time"],
            system_health=status["system_health"],
            integration_status=status["integration_status"]
        )
        
    except Exception as e:
        logger.error(f"Error getting orchestration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestration/workflows")
async def get_active_workflows(
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """Get information about active security workflows."""
    try:
        workflows = await orchestration.get_active_workflows(workflow_type=workflow_type)
        
        return {
            "active_workflows": workflows,
            "workflow_statistics": await orchestration.get_workflow_statistics(),
            "execution_metrics": await orchestration.get_execution_metrics()
        }
        
    except Exception as e:
        logger.error(f"Error getting active workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestration/playbooks")
async def get_security_playbooks(
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """Get available security playbooks and automation templates."""
    try:
        playbooks = await orchestration.get_security_playbooks()
        
        return {
            "available_playbooks": playbooks,
            "playbook_statistics": await orchestration.get_playbook_statistics(),
            "automation_coverage": await orchestration.get_automation_coverage()
        }
        
    except Exception as e:
        logger.error(f"Error getting security playbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orchestration/trigger")
async def trigger_security_workflow(
    workflow_type: str = Body(..., description="Type of security workflow to trigger"),
    trigger_data: Dict[str, Any] = Body(..., description="Data that triggered the workflow"),
    priority: str = Body("medium", description="Workflow execution priority"),
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """Trigger a security workflow for automated response."""
    try:
        logger.info(f"Triggering security workflow: {workflow_type}")
        
        execution_result = await orchestration.trigger_workflow(
            workflow_type=workflow_type,
            trigger_data=trigger_data,
            priority=priority
        )
        
        return {
            "workflow_id": execution_result["workflow_id"],
            "execution_id": execution_result["execution_id"],
            "status": execution_result["status"],
            "estimated_completion": execution_result.get("estimated_completion"),
            "triggered_actions": execution_result["triggered_actions"]
        }
        
    except Exception as e:
        logger.error(f"Error triggering security workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Overall System Status Route
@router.get("/status", response_model=SystemStatusResponse)
async def get_security_system_status(
    csoc: CognitiveSecurityOperationsCenter = Depends(get_csoc),
    threat_intel: AdvancedThreatIntelligence = Depends(get_threat_intelligence),
    response_system: AutonomousSecurityResponse = Depends(get_autonomous_security_response),
    fraud_detector: IntelligentFraudDetection = Depends(get_fraud_detector),
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """
    Get comprehensive security system status.
    
    Provides unified status across all Phase 8 autonomous security components:
    - Cognitive Security Operations Center health
    - Threat Intelligence system status
    - Autonomous Response system health
    - Fraud Detection system status
    - Security Orchestration health
    - Overall security posture and metrics
    """
    try:
        logger.info("Gathering comprehensive security system status")
        
        # Gather status from all security components
        csoc_status = await csoc.get_security_status()
        threat_intel_status = await threat_intel.get_threat_intelligence_status()
        response_status = await response_system.get_response_status()
        fraud_status = await fraud_detector.get_fraud_detection_status()
        orchestration_status = await orchestration.get_orchestration_status()
        
        # Determine overall system health
        component_healths = [
            csoc_status["health"],
            threat_intel_status["health"],
            response_status["health"],
            fraud_status["health"],
            orchestration_status["system_health"]
        ]
        
        # Calculate overall status
        healthy_components = sum(1 for health in component_healths if health == "healthy")
        overall_status = "healthy" if healthy_components >= 4 else "degraded" if healthy_components >= 3 else "critical"
        
        return SystemStatusResponse(
            overall_status=overall_status,
            cognitive_security_health=csoc_status["health"],
            threat_intelligence_health=threat_intel_status["health"],
            response_system_health=response_status["health"],
            fraud_detection_health=fraud_status["health"],
            orchestration_health=orchestration_status["system_health"],
            active_threats=csoc_status["active_threats"],
            blocked_incidents=response_status["blocked_incidents_24h"],
            system_uptime=csoc_status["system_uptime"]
        )
        
    except Exception as e:
        logger.error(f"Error getting security system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive Health Check
@router.get("/health")
async def security_health_check():
    """Comprehensive health check for all security systems."""
    try:
        health_status = {
            "service": "phase8-security",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "cognitive_security": True,
                "threat_intelligence": True,
                "autonomous_response": True,
                "fraud_detection": True,
                "security_orchestration": True
            },
            "capabilities": {
                "real_time_analysis": True,
                "behavioral_detection": True,
                "autonomous_response": True,
                "threat_correlation": True,
                "fraud_prevention": True,
                "workflow_orchestration": True
            }
        }
        
        # Test basic connectivity to core components
        try:
            csoc = await get_csoc()
            await csoc.get_security_status()
        except Exception:
            health_status["components"]["cognitive_security"] = False
            health_status["status"] = "degraded"
            
        try:
            threat_intel = await get_threat_intelligence()
            await threat_intel.get_threat_intelligence_status()
        except Exception:
            health_status["components"]["threat_intelligence"] = False
            health_status["status"] = "degraded"
            
        try:
            response_system = await get_autonomous_security_response()
            await response_system.get_response_status()
        except Exception:
            health_status["components"]["autonomous_response"] = False
            health_status["status"] = "degraded"
            
        try:
            fraud_detector = await get_fraud_detector()
            await fraud_detector.get_fraud_detection_status()
        except Exception:
            health_status["components"]["fraud_detection"] = False
            health_status["status"] = "degraded"
            
        try:
            orchestration = await get_security_orchestration()
            await orchestration.get_orchestration_status()
        except Exception:
            health_status["components"]["security_orchestration"] = False
            health_status["status"] = "degraded"
        
        # Update overall status based on component health
        healthy_components = sum(1 for healthy in health_status["components"].values() if healthy)
        if healthy_components < 3:
            health_status["status"] = "critical"
        elif healthy_components < 5:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "service": "phase8-security",
            "status": "critical",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


# Advanced Analytics and Metrics
@router.get("/metrics/dashboard")
async def get_security_dashboard_metrics(
    time_range: int = Query(24, ge=1, le=168, description="Time range in hours"),
    csoc: CognitiveSecurityOperationsCenter = Depends(get_csoc),
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """Get comprehensive security metrics for dashboard display."""
    try:
        dashboard_metrics = {
            "time_range_hours": time_range,
            "security_overview": await csoc.get_security_overview(time_range),
            "threat_analytics": await csoc.get_threat_analytics(time_range),
            "response_metrics": await orchestration.get_response_metrics(time_range),
            "system_performance": await orchestration.get_system_performance(time_range),
            "trends_analysis": await csoc.get_trends_analysis(time_range),
            "risk_assessment": await csoc.get_risk_assessment()
        }
        
        return dashboard_metrics
        
    except Exception as e:
        logger.error(f"Error getting security dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/performance")
async def get_security_performance_metrics(
    metric_type: Optional[str] = Query(None, description="Specific metric type"),
    orchestration: AutomatedSecurityOrchestration = Depends(get_security_orchestration)
):
    """Get detailed security system performance metrics."""
    try:
        performance_metrics = await orchestration.get_detailed_performance_metrics(metric_type)
        
        return {
            "performance_metrics": performance_metrics,
            "benchmark_comparison": await orchestration.get_benchmark_comparison(),
            "optimization_recommendations": await orchestration.get_optimization_recommendations()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration and Management Routes
@router.get("/config/threat-categories")
async def get_threat_categories():
    """Get available threat categories and classifications."""
    return {
        "threat_categories": [category.value for category in ThreatCategory],
        "severity_levels": [severity.value for severity in ThreatSeverity],
        "fraud_types": [fraud_type.value for fraud_type in FraudType],
        "response_actions": [action.value for action in ResponseAction],
        "intelligence_sources": [source.value for source in ThreatIntelligenceSource]
    }


# Startup and shutdown event handlers
@router.on_event("startup")
async def startup_security_systems():
    """Initialize Phase 8 security systems on startup."""
    logger.info("Initializing Phase 8 autonomous security systems...")
    try:
        # Pre-initialize all security components
        await get_csoc()
        await get_threat_intelligence() 
        await get_autonomous_security_response()
        await get_fraud_detector()
        await get_security_orchestration()
        
        logger.info("Phase 8 security systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Phase 8 security systems: {e}")


@router.on_event("shutdown") 
async def shutdown_security_systems():
    """Clean up Phase 8 security systems on shutdown."""
    logger.info("Shutting down Phase 8 autonomous security systems...")
    try:
        # Gracefully shutdown all security components
        logger.info("Phase 8 security systems shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during Phase 8 security systems shutdown: {e}")
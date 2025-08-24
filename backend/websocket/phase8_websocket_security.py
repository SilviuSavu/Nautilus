"""
Phase 8 WebSocket Security Streaming
Real-time Security Event Feeds for Autonomous Security Operations

Streams security events from Phase 8 components:
- Cognitive Security Operations Center (CSOC) alerts
- Intelligent Fraud Detection alerts
- Threat Intelligence updates
- Autonomous Security Response actions
- Security Orchestration workflow updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import asdict
import uuid

# Phase 8 Security Components - Using absolute imports
try:
    from phase8_autonomous_operations.security.cognitive_security_operations_center import (
        get_csoc, CognitiveAlert
    )
    from phase8_autonomous_operations.fraud_detection.intelligent_fraud_detection import (
        get_fraud_detector, FraudAlert
    )
    from phase8_autonomous_operations.security_orchestration.automated_security_orchestration import (
        get_security_orchestration, WorkflowExecution
    )
    from phase8_autonomous_operations.threat_intelligence.advanced_threat_intelligence import (
        get_threat_intelligence
    )
    from phase8_autonomous_operations.security_response.autonomous_security_response import (
        get_autonomous_security_response
    )
except ImportError as e:
    logging.warning(f"Phase 8 imports not available: {e}")
    # Fallback - will be handled gracefully in the code

logger = logging.getLogger(__name__)


class Phase8SecurityEventStreamer:
    """Streams real-time security events from Phase 8 components"""
    
    def __init__(self, websocket_manager, redis_client=None):
        self.websocket_manager = websocket_manager
        self.redis_client = redis_client
        self.active_subscriptions = {}
        self.event_publishers = {}
        self.security_components = {}
        self.streaming_tasks = {}
        
    async def initialize(self):
        """Initialize Phase 8 security components and event publishers"""
        try:
            # Initialize security components
            self.security_components = {
                'csoc': await get_csoc(),
                'fraud_detector': await get_fraud_detector(),
                'security_orchestration': await get_security_orchestration(),
                'threat_intelligence': await get_threat_intelligence(),
                'security_response': await get_autonomous_security_response()
            }
            
            # Initialize event publishers
            await self._initialize_event_publishers()
            
            logger.info("Phase 8 Security Event Streamer initialized")
            
        except Exception as e:
            logger.error(f"Phase 8 Security Streamer initialization failed: {e}")
            raise
    
    async def _initialize_event_publishers(self):
        """Initialize event publishers for each security component"""
        
        # CSOC Security Analysis Publisher
        self.event_publishers['security_analysis'] = SecurityAnalysisPublisher(
            self.websocket_manager, self.security_components['csoc']
        )
        
        # Fraud Detection Publisher
        self.event_publishers['fraud_detection'] = FraudDetectionPublisher(
            self.websocket_manager, self.security_components['fraud_detector']
        )
        
        # Threat Intelligence Publisher
        self.event_publishers['threat_intelligence'] = ThreatIntelligencePublisher(
            self.websocket_manager, self.security_components['threat_intelligence']
        )
        
        # Security Response Publisher
        self.event_publishers['security_response'] = SecurityResponsePublisher(
            self.websocket_manager, self.security_components['security_response']
        )
        
        # Security Orchestration Publisher
        self.event_publishers['security_orchestration'] = SecurityOrchestrationPublisher(
            self.websocket_manager, self.security_components['security_orchestration']
        )
        
        # Initialize all publishers
        for publisher in self.event_publishers.values():
            await publisher.initialize()
    
    async def subscribe_to_security_events(self, connection_id: str, 
                                         event_types: Optional[List[str]] = None):
        """Subscribe connection to security event streams"""
        try:
            if event_types is None:
                event_types = [
                    'security.analysis.complete',
                    'security.threat.detected', 
                    'security.fraud.alert',
                    'security.response.executed',
                    'security.orchestration.update'
                ]
            
            # Store subscription preferences
            self.active_subscriptions[connection_id] = {
                'event_types': set(event_types),
                'subscribed_at': datetime.now(),
                'filter_criteria': {}
            }
            
            # Subscribe to relevant topics in WebSocket manager
            for event_type in event_types:
                topic = f"phase8_security_{event_type.replace('.', '_')}"
                self.websocket_manager.subscribe_to_topic(connection_id, topic)
            
            # Start streaming for this connection if not already started
            if connection_id not in self.streaming_tasks:
                self.streaming_tasks[connection_id] = asyncio.create_task(
                    self._stream_security_events(connection_id)
                )
            
            logger.info(f"Connection {connection_id} subscribed to {len(event_types)} security event types")
            
        except Exception as e:
            logger.error(f"Security event subscription failed: {e}")
    
    async def unsubscribe_from_security_events(self, connection_id: str, 
                                             event_types: Optional[List[str]] = None):
        """Unsubscribe connection from security event streams"""
        try:
            if connection_id not in self.active_subscriptions:
                return
            
            if event_types is None:
                # Unsubscribe from all
                for event_type in self.active_subscriptions[connection_id]['event_types']:
                    topic = f"phase8_security_{event_type.replace('.', '_')}"
                    self.websocket_manager.unsubscribe_from_topic(connection_id, topic)
                
                # Cancel streaming task
                if connection_id in self.streaming_tasks:
                    self.streaming_tasks[connection_id].cancel()
                    del self.streaming_tasks[connection_id]
                
                del self.active_subscriptions[connection_id]
                
            else:
                # Unsubscribe from specific event types
                for event_type in event_types:
                    if event_type in self.active_subscriptions[connection_id]['event_types']:
                        self.active_subscriptions[connection_id]['event_types'].remove(event_type)
                        topic = f"phase8_security_{event_type.replace('.', '_')}"
                        self.websocket_manager.unsubscribe_from_topic(connection_id, topic)
            
            logger.info(f"Connection {connection_id} unsubscribed from security events")
            
        except Exception as e:
            logger.error(f"Security event unsubscription failed: {e}")
    
    async def _stream_security_events(self, connection_id: str):
        """Stream security events to specific connection"""
        try:
            while connection_id in self.active_subscriptions:
                # Check for new security events from all publishers
                for event_type, publisher in self.event_publishers.items():
                    if await self._should_stream_event_type(connection_id, event_type):
                        latest_events = await publisher.get_latest_events()
                        
                        for event in latest_events:
                            await self._send_security_event(connection_id, event)
                
                # Stream at high frequency for real-time updates
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info(f"Security event streaming cancelled for connection: {connection_id}")
        except Exception as e:
            logger.error(f"Security event streaming error for {connection_id}: {e}")
    
    async def _should_stream_event_type(self, connection_id: str, event_type: str) -> bool:
        """Check if connection should receive this event type"""
        if connection_id not in self.active_subscriptions:
            return False
        
        subscription = self.active_subscriptions[connection_id]
        
        # Check if event type is subscribed
        event_type_key = f"security.{event_type}.update"
        return event_type_key in subscription['event_types']
    
    async def _send_security_event(self, connection_id: str, event: Dict[str, Any]):
        """Send security event to specific connection"""
        try:
            # Add metadata
            event_with_metadata = {
                "type": "security_event",
                "data": event,
                "timestamp": datetime.now().isoformat(),
                "source": "phase8_security",
                "connection_id": connection_id
            }
            
            await self.websocket_manager.send_personal_message(
                event_with_metadata, connection_id
            )
            
        except Exception as e:
            logger.error(f"Failed to send security event: {e}")
    
    async def cleanup_connection(self, connection_id: str):
        """Cleanup when connection is closed"""
        try:
            await self.unsubscribe_from_security_events(connection_id)
            
        except Exception as e:
            logger.error(f"Connection cleanup failed: {e}")


class SecurityAnalysisPublisher:
    """Publisher for CSOC security analysis events"""
    
    def __init__(self, websocket_manager, csoc):
        self.websocket_manager = websocket_manager
        self.csoc = csoc
        self.last_check = datetime.now()
        self.processed_alerts = set()
        
    async def initialize(self):
        """Initialize security analysis publisher"""
        logger.info("Security Analysis Publisher initialized")
        
    async def get_latest_events(self) -> List[Dict[str, Any]]:
        """Get latest security analysis events"""
        try:
            events = []
            
            # Get current security status from CSOC
            if self.csoc and self.csoc.active_monitoring:
                security_status = await self.csoc.get_security_status()
                
                # Check for new alerts
                if 'active_threats' in security_status and security_status['active_threats'] > 0:
                    # Mock getting specific alert details - in production this would
                    # access the actual alert data from CSOC
                    new_analysis_event = {
                        "event_type": "security.analysis.complete",
                        "alert_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "analysis_results": {
                            "threat_detected": True,
                            "confidence_score": 0.87,
                            "threat_type": "behavioral_anomaly",
                            "affected_systems": ["trading_engine", "user_session"],
                            "risk_level": "medium",
                            "recommended_actions": [
                                "Monitor user session closely",
                                "Verify trading patterns",
                                "Check for unauthorized access"
                            ]
                        },
                        "cognitive_insights": {
                            "behavioral_deviation": 0.65,
                            "pattern_match": "anomalous_trading_frequency",
                            "ml_prediction": "potential_account_compromise"
                        }
                    }
                    
                    if new_analysis_event["alert_id"] not in self.processed_alerts:
                        events.append(new_analysis_event)
                        self.processed_alerts.add(new_analysis_event["alert_id"])
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get security analysis events: {e}")
            return []


class FraudDetectionPublisher:
    """Publisher for fraud detection events"""
    
    def __init__(self, websocket_manager, fraud_detector):
        self.websocket_manager = websocket_manager
        self.fraud_detector = fraud_detector
        self.last_alert_check = datetime.now()
        self.processed_alerts = set()
        
    async def initialize(self):
        """Initialize fraud detection publisher"""
        logger.info("Fraud Detection Publisher initialized")
        
    async def get_latest_events(self) -> List[Dict[str, Any]]:
        """Get latest fraud detection events"""
        try:
            events = []
            
            if self.fraud_detector and self.fraud_detector.active:
                # Check for new fraud alerts from the alert queue
                if hasattr(self.fraud_detector, 'alert_queue') and self.fraud_detector.alert_queue:
                    # Get recent alerts
                    recent_alerts = list(self.fraud_detector.alert_queue)[-10:]  # Last 10 alerts
                    
                    for alert in recent_alerts:
                        if isinstance(alert, FraudAlert):
                            alert_key = f"{alert.alert_id}_{alert.timestamp.isoformat()}"
                            
                            if alert_key not in self.processed_alerts:
                                fraud_event = {
                                    "event_type": "security.fraud.alert",
                                    "alert_id": alert.alert_id,
                                    "timestamp": alert.timestamp.isoformat(),
                                    "fraud_details": {
                                        "fraud_type": alert.fraud_type.value,
                                        "severity": alert.severity.value,
                                        "confidence_score": alert.confidence_score,
                                        "risk_score": alert.risk_score,
                                        "detection_model": alert.detection_model.value,
                                        "affected_entities": alert.affected_entities,
                                        "financial_impact": alert.financial_impact,
                                        "investigation_priority": alert.investigation_priority
                                    },
                                    "behavioral_indicators": alert.behavioral_indicators,
                                    "recommended_actions": alert.recommended_actions,
                                    "false_positive_likelihood": alert.false_positive_likelihood
                                }
                                
                                events.append(fraud_event)
                                self.processed_alerts.add(alert_key)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get fraud detection events: {e}")
            return []


class ThreatIntelligencePublisher:
    """Publisher for threat intelligence events"""
    
    def __init__(self, websocket_manager, threat_intelligence):
        self.websocket_manager = websocket_manager
        self.threat_intelligence = threat_intelligence
        self.last_intelligence_check = datetime.now()
        self.processed_indicators = set()
        
    async def initialize(self):
        """Initialize threat intelligence publisher"""
        logger.info("Threat Intelligence Publisher initialized")
        
    async def get_latest_events(self) -> List[Dict[str, Any]]:
        """Get latest threat intelligence events"""
        try:
            events = []
            
            # Simulate threat intelligence updates
            # In production, this would pull from actual threat intelligence feeds
            current_time = datetime.now()
            
            # Generate periodic threat intelligence updates
            if (current_time - self.last_intelligence_check).seconds > 30:
                threat_event = {
                    "event_type": "security.threat.detected",
                    "indicator_id": str(uuid.uuid4()),
                    "timestamp": current_time.isoformat(),
                    "threat_intelligence": {
                        "indicator_type": "ip_address",
                        "indicator_value": "192.168.1.100",
                        "threat_type": "malicious_activity",
                        "confidence": 0.82,
                        "severity": "medium",
                        "source": "external_feed",
                        "first_seen": current_time.isoformat(),
                        "tags": ["botnet", "trading_platform_targeting"],
                        "geolocation": {
                            "country": "Unknown",
                            "region": "Unknown"
                        }
                    },
                    "context": {
                        "related_campaigns": ["APT-Trading-2024"],
                        "attack_techniques": ["credential_stuffing", "api_abuse"],
                        "affected_sectors": ["financial_services", "trading"]
                    }
                }
                
                indicator_key = f"{threat_event['indicator_id']}"
                if indicator_key not in self.processed_indicators:
                    events.append(threat_event)
                    self.processed_indicators.add(indicator_key)
                
                self.last_intelligence_check = current_time
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get threat intelligence events: {e}")
            return []


class SecurityResponsePublisher:
    """Publisher for autonomous security response events"""
    
    def __init__(self, websocket_manager, security_response):
        self.websocket_manager = websocket_manager
        self.security_response = security_response
        self.last_response_check = datetime.now()
        self.processed_responses = set()
        
    async def initialize(self):
        """Initialize security response publisher"""
        logger.info("Security Response Publisher initialized")
        
    async def get_latest_events(self) -> List[Dict[str, Any]]:
        """Get latest security response events"""
        try:
            events = []
            
            if self.security_response and self.security_response.active:
                # Check for recent response actions
                if hasattr(self.security_response, 'response_history'):
                    recent_responses = list(self.security_response.response_history)[-5:]
                    
                    for response in recent_responses:
                        response_key = f"{response.get('response_id', str(uuid.uuid4()))}_{response.get('timestamp', datetime.now().isoformat())}"
                        
                        if response_key not in self.processed_responses:
                            response_event = {
                                "event_type": "security.response.executed",
                                "response_id": response.get('response_id', str(uuid.uuid4())),
                                "timestamp": response.get('timestamp', datetime.now().isoformat()),
                                "response_details": {
                                    "action_type": response.get('action_type', 'quarantine_user'),
                                    "target_entity": response.get('target', 'user_session'),
                                    "severity": response.get('severity', 'medium'),
                                    "execution_status": response.get('status', 'completed'),
                                    "response_time_ms": response.get('response_time', 150),
                                    "effectiveness": response.get('effectiveness', 0.9)
                                },
                                "trigger_event": response.get('trigger_event', {}),
                                "mitigation_steps": response.get('mitigation_steps', [
                                    "User session terminated",
                                    "Access credentials revoked",
                                    "Security team notified"
                                ]),
                                "follow_up_required": response.get('follow_up_required', False)
                            }
                            
                            events.append(response_event)
                            self.processed_responses.add(response_key)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get security response events: {e}")
            return []


class SecurityOrchestrationPublisher:
    """Publisher for security orchestration workflow events"""
    
    def __init__(self, websocket_manager, security_orchestration):
        self.websocket_manager = websocket_manager
        self.security_orchestration = security_orchestration
        self.last_workflow_check = datetime.now()
        self.processed_workflows = set()
        
    async def initialize(self):
        """Initialize security orchestration publisher"""
        logger.info("Security Orchestration Publisher initialized")
        
    async def get_latest_events(self) -> List[Dict[str, Any]]:
        """Get latest security orchestration events"""
        try:
            events = []
            
            if self.security_orchestration:
                # Get orchestration status
                orchestration_status = await self.security_orchestration.get_orchestration_status()
                
                # Check for workflow updates
                if orchestration_status.get('active_executions', 0) > 0:
                    # Simulate workflow execution update
                    workflow_event = {
                        "event_type": "security.orchestration.update",
                        "workflow_execution_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "orchestration_details": {
                            "workflow_id": "incident_response_high_severity",
                            "workflow_name": "High Severity Incident Response",
                            "execution_status": "running",
                            "current_step": "threat_analysis",
                            "steps_completed": 1,
                            "total_steps": 3,
                            "execution_time_seconds": 45,
                            "estimated_completion": (datetime.now() + 
                                                   timedelta(minutes=5)).isoformat()
                        },
                        "workflow_context": {
                            "trigger_event_type": "security_alert", 
                            "severity": "high",
                            "affected_systems": ["trading_engine"],
                            "orchestration_state": orchestration_status.get('state', 'monitoring')
                        },
                        "step_results": [
                            {
                                "step_name": "threat_analysis",
                                "status": "completed",
                                "result": "Threat confirmed with 87% confidence",
                                "execution_time_ms": 2500
                            }
                        ]
                    }
                    
                    workflow_key = workflow_event["workflow_execution_id"]
                    if workflow_key not in self.processed_workflows:
                        events.append(workflow_event)
                        self.processed_workflows.add(workflow_key)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get orchestration events: {e}")
            return []


# Global Phase 8 Security Event Streamer
phase8_security_streamer = None

async def get_phase8_security_streamer(websocket_manager, redis_client=None):
    """Get or create Phase 8 security event streamer"""
    global phase8_security_streamer
    if not phase8_security_streamer:
        phase8_security_streamer = Phase8SecurityEventStreamer(websocket_manager, redis_client)
        await phase8_security_streamer.initialize()
    return phase8_security_streamer


async def handle_phase8_security_subscription(connection_id: str, message: Dict[str, Any],
                                            websocket_manager) -> Dict[str, Any]:
    """Handle Phase 8 security event subscription requests"""
    try:
        streamer = await get_phase8_security_streamer(websocket_manager)
        
        action = message.get("action")
        event_types = message.get("event_types")
        
        if action == "subscribe":
            await streamer.subscribe_to_security_events(connection_id, event_types)
            return {
                "success": True,
                "message": "Subscribed to Phase 8 security events",
                "subscribed_events": event_types or [
                    "security.analysis.complete",
                    "security.threat.detected", 
                    "security.fraud.alert",
                    "security.response.executed",
                    "security.orchestration.update"
                ]
            }
        elif action == "unsubscribe":
            await streamer.unsubscribe_from_security_events(connection_id, event_types)
            return {
                "success": True,
                "message": "Unsubscribed from Phase 8 security events"
            }
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
            
    except Exception as e:
        logger.error(f"Phase 8 security subscription handling failed: {e}")
        return {
            "success": False, 
            "error": str(e)
        }


async def cleanup_phase8_security_connection(connection_id: str):
    """Cleanup Phase 8 security streaming for connection"""
    try:
        if phase8_security_streamer:
            await phase8_security_streamer.cleanup_connection(connection_id)
            
    except Exception as e:
        logger.error(f"Phase 8 security connection cleanup failed: {e}")


# Phase 8 Security Event Types
PHASE8_SECURITY_EVENT_TYPES = [
    "security.analysis.complete",
    "security.threat.detected", 
    "security.fraud.alert",
    "security.response.executed",
    "security.orchestration.update"
]


def get_phase8_security_event_schema() -> Dict[str, Any]:
    """Get schema for Phase 8 security events"""
    return {
        "security.analysis.complete": {
            "description": "Cognitive security analysis completed",
            "fields": {
                "analysis_results": "Security analysis results with threat detection",
                "cognitive_insights": "AI-driven behavioral and pattern insights",
                "confidence_score": "Analysis confidence (0.0-1.0)",
                "risk_level": "Assessed risk level (low/medium/high/critical)"
            }
        },
        "security.threat.detected": {
            "description": "New threat intelligence detected",
            "fields": {
                "threat_intelligence": "Threat indicator details and metadata",
                "context": "Related campaigns and attack techniques",
                "confidence": "Threat confidence score (0.0-1.0)",
                "severity": "Threat severity level"
            }
        },
        "security.fraud.alert": {
            "description": "Fraud detection alert generated",
            "fields": {
                "fraud_details": "Fraud type, severity, and detection model",
                "behavioral_indicators": "Behavioral anomalies detected",
                "financial_impact": "Estimated financial impact",
                "investigation_priority": "Priority score (1-10)"
            }
        },
        "security.response.executed": {
            "description": "Autonomous security response executed",
            "fields": {
                "response_details": "Response action and execution status",
                "mitigation_steps": "Steps taken to mitigate threat",
                "effectiveness": "Response effectiveness score",
                "follow_up_required": "Whether manual follow-up is needed"
            }
        },
        "security.orchestration.update": {
            "description": "Security orchestration workflow status update",
            "fields": {
                "orchestration_details": "Workflow execution status and progress",
                "workflow_context": "Trigger event and orchestration state",
                "step_results": "Individual workflow step results",
                "estimated_completion": "Expected workflow completion time"
            }
        }
    }
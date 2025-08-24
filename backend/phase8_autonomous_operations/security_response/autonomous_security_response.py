"""
Autonomous Security Response System
Adaptive Countermeasures & Intelligent Response for Financial Trading Platforms

Provides real-time autonomous security response with adaptive countermeasures,
intelligent threat mitigation, and automated incident response workflows.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp

logger = logging.getLogger(__name__)


class ResponseAction(Enum):
    """Types of autonomous security responses"""
    ALERT_ONLY = "alert_only"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    SUSPEND_USER = "suspend_user"
    QUARANTINE_SESSION = "quarantine_session"
    FORCE_REAUTHENTICATION = "force_reauth"
    NETWORK_ISOLATION = "network_isolation"
    API_THROTTLING = "api_throttling"
    EMERGENCY_STOP = "emergency_stop"
    ROLLBACK_CHANGES = "rollback_changes"
    HONEYPOT_REDIRECT = "honeypot_redirect"
    DATA_ENCRYPTION = "data_encryption"
    BACKUP_ACTIVATION = "backup_activation"
    INCIDENT_ESCALATION = "incident_escalation"


class ResponseSeverity(Enum):
    """Response severity levels"""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AdaptationStrategy(Enum):
    """Response adaptation strategies"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    LEARNING = "learning"
    PREDICTIVE = "predictive"
    COLLABORATIVE = "collaborative"


@dataclass
class SecurityResponse:
    """Autonomous security response definition"""
    response_id: str
    trigger_conditions: Dict[str, Any]
    actions: List[ResponseAction]
    severity: ResponseSeverity
    confidence_threshold: float
    execution_delay: int  # seconds
    duration: Optional[int] = None  # seconds, None for permanent
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.STATIC
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    rollback_actions: List[ResponseAction] = field(default_factory=list)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    effectiveness_score: float = 0.8
    last_executed: Optional[datetime] = None
    execution_count: int = 0


@dataclass
class ResponseExecution:
    """Record of response execution"""
    execution_id: str
    response_id: str
    triggered_by: str  # Alert ID or event ID
    trigger_time: datetime
    execution_time: datetime
    completion_time: Optional[datetime]
    status: str  # pending, executing, completed, failed, rolled_back
    actions_executed: List[str]
    effectiveness: Optional[float] = None
    side_effects_observed: List[str] = field(default_factory=list)
    rollback_required: bool = False
    rollback_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveCountermeasure:
    """Self-adapting security countermeasure"""
    countermeasure_id: str
    name: str
    target_threat_types: List[str]
    base_parameters: Dict[str, Any]
    current_parameters: Dict[str, Any]
    adaptation_history: List[Dict[str, Any]]
    effectiveness_trend: List[float]
    learning_rate: float = 0.1
    adaptation_bounds: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    last_adapted: Optional[datetime] = None
    adaptation_count: int = 0
    locked: bool = False  # Prevent adaptation if true


class ResponseOrchestrator:
    """Orchestrates autonomous security responses"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.active_responses = {}
        self.response_catalog = {}
        self.execution_history = deque(maxlen=10000)
        self.adaptation_engine = AdaptationEngine()
        self.escalation_chains = {}
        
    async def register_response(self, response: SecurityResponse):
        """Register a security response for autonomous execution"""
        self.response_catalog[response.response_id] = response
        
        # Store in Redis for persistence
        response_key = f"security_response:{response.response_id}"
        await self.redis.hset(response_key, mapping=response.__dict__)
        
        logger.info(f"Registered security response: {response.response_id}")
    
    async def evaluate_response_triggers(self, event_data: Dict[str, Any]) -> List[SecurityResponse]:
        """Evaluate which responses should be triggered by an event"""
        triggered_responses = []
        
        for response in self.response_catalog.values():
            if await self._should_trigger_response(response, event_data):
                triggered_responses.append(response)
        
        # Sort by severity and confidence
        triggered_responses.sort(
            key=lambda r: (r.severity.value, r.confidence_threshold),
            reverse=True
        )
        
        return triggered_responses
    
    async def _should_trigger_response(self, response: SecurityResponse, 
                                     event_data: Dict[str, Any]) -> bool:
        """Check if response should be triggered by event"""
        try:
            conditions = response.trigger_conditions
            
            # Check event type match
            if "event_type" in conditions:
                if event_data.get("event_type") not in conditions["event_type"]:
                    return False
            
            # Check severity threshold
            if "min_severity" in conditions:
                event_severity = event_data.get("severity", "low")
                if not self._severity_meets_threshold(event_severity, conditions["min_severity"]):
                    return False
            
            # Check confidence threshold
            if "min_confidence" in conditions:
                event_confidence = event_data.get("confidence", 0.0)
                if event_confidence < conditions["min_confidence"]:
                    return False
            
            # Check threat type match
            if "threat_types" in conditions:
                event_threat_type = event_data.get("threat_type")
                if event_threat_type not in conditions["threat_types"]:
                    return False
            
            # Check affected systems
            if "affected_systems" in conditions:
                event_systems = set(event_data.get("affected_systems", []))
                required_systems = set(conditions["affected_systems"])
                if not event_systems.intersection(required_systems):
                    return False
            
            # Check rate limiting (don't trigger too frequently)
            if await self._is_rate_limited(response):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response trigger evaluation failed: {e}")
            return False
    
    def _severity_meets_threshold(self, event_severity: str, threshold: str) -> bool:
        """Check if event severity meets response threshold"""
        severity_levels = {
            "low": 1,
            "medium": 2, 
            "high": 3,
            "critical": 4,
            "emergency": 5
        }
        
        event_level = severity_levels.get(event_severity, 0)
        threshold_level = severity_levels.get(threshold, 5)
        
        return event_level >= threshold_level
    
    async def execute_response(self, response: SecurityResponse, 
                             trigger_event: Dict[str, Any]) -> ResponseExecution:
        """Execute autonomous security response"""
        execution = ResponseExecution(
            execution_id=str(uuid.uuid4()),
            response_id=response.response_id,
            triggered_by=trigger_event.get("event_id", "unknown"),
            trigger_time=datetime.now(),
            execution_time=datetime.now(),
            status="pending",
            actions_executed=[]
        )
        
        try:
            # Add execution delay if specified
            if response.execution_delay > 0:
                await asyncio.sleep(response.execution_delay)
            
            execution.status = "executing"
            
            # Execute each action in the response
            for action in response.actions:
                success = await self._execute_action(action, trigger_event, execution)
                if success:
                    execution.actions_executed.append(action.value)
                else:
                    logger.warning(f"Action {action.value} failed in response {response.response_id}")
            
            # Set response duration if specified
            if response.duration:
                await asyncio.sleep(response.duration)
                await self._rollback_response(response, execution)
            
            execution.completion_time = datetime.now()
            execution.status = "completed"
            
            # Update response statistics
            response.last_executed = datetime.now()
            response.execution_count += 1
            
            # Evaluate effectiveness
            await self._evaluate_response_effectiveness(response, execution)
            
            # Store execution record
            await self._store_execution_record(execution)
            
            self.execution_history.append(execution)
            
            logger.info(f"Security response {response.response_id} executed successfully")
            
        except Exception as e:
            execution.status = "failed"
            execution.completion_time = datetime.now()
            logger.error(f"Response execution failed: {e}")
        
        return execution
    
    async def _execute_action(self, action: ResponseAction, 
                            trigger_event: Dict[str, Any],
                            execution: ResponseExecution) -> bool:
        """Execute individual security action"""
        try:
            if action == ResponseAction.BLOCK_IP:
                return await self._block_ip_address(trigger_event.get("source_ip"))
            
            elif action == ResponseAction.SUSPEND_USER:
                return await self._suspend_user(trigger_event.get("user_id"))
            
            elif action == ResponseAction.RATE_LIMIT:
                return await self._apply_rate_limit(trigger_event)
            
            elif action == ResponseAction.QUARANTINE_SESSION:
                return await self._quarantine_session(trigger_event.get("session_id"))
            
            elif action == ResponseAction.FORCE_REAUTHENTICATION:
                return await self._force_reauthentication(trigger_event.get("user_id"))
            
            elif action == ResponseAction.API_THROTTLING:
                return await self._apply_api_throttling(trigger_event)
            
            elif action == ResponseAction.EMERGENCY_STOP:
                return await self._emergency_stop(trigger_event)
            
            elif action == ResponseAction.HONEYPOT_REDIRECT:
                return await self._redirect_to_honeypot(trigger_event)
            
            elif action == ResponseAction.INCIDENT_ESCALATION:
                return await self._escalate_incident(trigger_event, execution)
            
            else:
                logger.warning(f"Unknown action type: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {action.value}: {e}")
            return False
    
    async def _block_ip_address(self, ip_address: str) -> bool:
        """Block IP address at network level"""
        if not ip_address:
            return False
        
        try:
            # Add IP to Redis blocked list
            await self.redis.sadd("blocked_ips", ip_address)
            await self.redis.expire("blocked_ips", 86400)  # 24 hours
            
            # Publish to security systems
            await self.redis.publish("security_actions", json.dumps({
                "action": "block_ip",
                "ip_address": ip_address,
                "timestamp": datetime.now().isoformat()
            }))
            
            logger.info(f"Blocked IP address: {ip_address}")
            return True
            
        except Exception as e:
            logger.error(f"IP blocking failed: {e}")
            return False
    
    async def _suspend_user(self, user_id: str) -> bool:
        """Suspend user account"""
        if not user_id:
            return False
        
        try:
            # Add user to suspended list
            await self.redis.sadd("suspended_users", user_id)
            
            # Invalidate user sessions
            await self.redis.delete(f"user_sessions:{user_id}")
            
            # Publish suspension event
            await self.redis.publish("user_actions", json.dumps({
                "action": "suspend_user",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            logger.info(f"Suspended user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"User suspension failed: {e}")
            return False
    
    async def _apply_rate_limit(self, trigger_event: Dict[str, Any]) -> bool:
        """Apply rate limiting based on event context"""
        try:
            source_ip = trigger_event.get("source_ip")
            user_id = trigger_event.get("user_id")
            
            if source_ip:
                # Apply IP-based rate limit
                rate_limit_key = f"rate_limit:ip:{source_ip}"
                await self.redis.setex(rate_limit_key, 3600, "100")  # 100 requests per hour
            
            if user_id:
                # Apply user-based rate limit
                rate_limit_key = f"rate_limit:user:{user_id}"
                await self.redis.setex(rate_limit_key, 3600, "50")  # 50 requests per hour
            
            logger.info(f"Applied rate limiting for event: {trigger_event.get('event_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting failed: {e}")
            return False
    
    async def _emergency_stop(self, trigger_event: Dict[str, Any]) -> bool:
        """Execute emergency stop procedures"""
        try:
            # Publish emergency stop signal
            await self.redis.publish("emergency_stop", json.dumps({
                "timestamp": datetime.now().isoformat(),
                "trigger_event": trigger_event,
                "severity": "critical"
            }))
            
            # Stop critical trading operations
            await self.redis.set("emergency_stop_active", "true")
            await self.redis.expire("emergency_stop_active", 1800)  # 30 minutes
            
            logger.critical("Emergency stop executed")
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False


class AdaptationEngine:
    """Engine for adapting security responses based on effectiveness"""
    
    def __init__(self):
        self.adaptation_strategies = {}
        self.learning_models = {}
        self.effectiveness_history = defaultdict(list)
        
    async def adapt_response(self, response: SecurityResponse, 
                           execution: ResponseExecution,
                           effectiveness_score: float):
        """Adapt response based on execution effectiveness"""
        try:
            if response.adaptation_strategy == AdaptationStrategy.STATIC:
                return  # No adaptation for static responses
            
            # Record effectiveness
            self.effectiveness_history[response.response_id].append(effectiveness_score)
            
            # Determine adaptation needed
            adaptation_needed = await self._evaluate_adaptation_need(
                response, effectiveness_score
            )
            
            if adaptation_needed:
                await self._apply_adaptation(response, execution, effectiveness_score)
            
        except Exception as e:
            logger.error(f"Response adaptation failed: {e}")
    
    async def _evaluate_adaptation_need(self, response: SecurityResponse, 
                                      effectiveness: float) -> bool:
        """Evaluate if response needs adaptation"""
        # Check if effectiveness is below threshold
        if effectiveness < 0.6:
            return True
        
        # Check effectiveness trend
        history = self.effectiveness_history[response.response_id]
        if len(history) >= 5:
            recent_avg = sum(history[-5:]) / 5
            older_avg = sum(history[-10:-5]) / 5 if len(history) >= 10 else recent_avg
            
            if recent_avg < older_avg - 0.1:  # Declining effectiveness
                return True
        
        return False
    
    async def _apply_adaptation(self, response: SecurityResponse,
                              execution: ResponseExecution,
                              effectiveness: float):
        """Apply adaptation to improve response effectiveness"""
        try:
            if response.adaptation_strategy == AdaptationStrategy.LEARNING:
                await self._apply_learning_adaptation(response, effectiveness)
            elif response.adaptation_strategy == AdaptationStrategy.DYNAMIC:
                await self._apply_dynamic_adaptation(response, execution)
            elif response.adaptation_strategy == AdaptationStrategy.PREDICTIVE:
                await self._apply_predictive_adaptation(response)
            
            logger.info(f"Applied adaptation to response: {response.response_id}")
            
        except Exception as e:
            logger.error(f"Adaptation application failed: {e}")


class AutonomousSecurityResponse:
    """
    Main autonomous security response system
    """
    
    def __init__(self, redis_url: str, database_url: str):
        self.redis_url = redis_url
        self.database_url = database_url
        self.redis = None
        self.orchestrator = None
        self.active = False
        self.response_metrics = defaultdict(int)
        self.countermeasures = {}
        
    async def initialize(self):
        """Initialize autonomous security response system"""
        try:
            # Initialize Redis connection
            self.redis = await redis.from_url(self.redis_url)
            
            # Initialize orchestrator
            self.orchestrator = ResponseOrchestrator(self.redis)
            
            # Load default responses
            await self._load_default_responses()
            
            # Load adaptive countermeasures
            await self._load_countermeasures()
            
            self.active = True
            
            logger.info("Autonomous Security Response system initialized")
            
        except Exception as e:
            logger.error(f"ASR initialization failed: {e}")
            raise
    
    async def _load_default_responses(self):
        """Load default security responses"""
        default_responses = [
            SecurityResponse(
                response_id="high_confidence_ip_block",
                trigger_conditions={
                    "min_confidence": 0.9,
                    "min_severity": "high",
                    "threat_types": ["unauthorized_access", "brute_force"]
                },
                actions=[ResponseAction.BLOCK_IP, ResponseAction.INCIDENT_ESCALATION],
                severity=ResponseSeverity.HIGH,
                confidence_threshold=0.9,
                execution_delay=0,
                duration=3600  # 1 hour
            ),
            SecurityResponse(
                response_id="suspicious_trading_activity",
                trigger_conditions={
                    "event_type": ["trading_anomaly", "market_manipulation"],
                    "min_confidence": 0.7
                },
                actions=[ResponseAction.RATE_LIMIT, ResponseAction.FORCE_REAUTHENTICATION],
                severity=ResponseSeverity.MEDIUM,
                confidence_threshold=0.7,
                execution_delay=5,
                adaptation_strategy=AdaptationStrategy.LEARNING
            ),
            SecurityResponse(
                response_id="emergency_threat",
                trigger_conditions={
                    "min_severity": "critical",
                    "min_confidence": 0.95
                },
                actions=[ResponseAction.EMERGENCY_STOP, ResponseAction.INCIDENT_ESCALATION],
                severity=ResponseSeverity.EMERGENCY,
                confidence_threshold=0.95,
                execution_delay=0
            )
        ]
        
        for response in default_responses:
            await self.orchestrator.register_response(response)
        
        logger.info(f"Loaded {len(default_responses)} default security responses")
    
    async def process_security_event(self, event_data: Dict[str, Any]) -> List[ResponseExecution]:
        """Process security event and execute appropriate responses"""
        if not self.active:
            return []
        
        try:
            # Evaluate which responses should be triggered
            triggered_responses = await self.orchestrator.evaluate_response_triggers(event_data)
            
            if not triggered_responses:
                return []
            
            # Execute responses
            executions = []
            for response in triggered_responses:
                execution = await self.orchestrator.execute_response(response, event_data)
                executions.append(execution)
                
                # Update metrics
                self.response_metrics[f"response_{response.response_id}"] += 1
                self.response_metrics[f"action_total"] += len(execution.actions_executed)
            
            logger.info(f"Executed {len(executions)} security responses for event")
            return executions
            
        except Exception as e:
            logger.error(f"Security event processing failed: {e}")
            return []
    
    async def add_custom_response(self, response_config: Dict[str, Any]) -> bool:
        """Add custom security response"""
        try:
            response = SecurityResponse(
                response_id=response_config["response_id"],
                trigger_conditions=response_config["trigger_conditions"],
                actions=[ResponseAction(a) for a in response_config["actions"]],
                severity=ResponseSeverity(response_config["severity"]),
                confidence_threshold=response_config["confidence_threshold"],
                execution_delay=response_config.get("execution_delay", 0),
                duration=response_config.get("duration"),
                adaptation_strategy=AdaptationStrategy(
                    response_config.get("adaptation_strategy", "static")
                )
            )
            
            await self.orchestrator.register_response(response)
            return True
            
        except Exception as e:
            logger.error(f"Custom response registration failed: {e}")
            return False
    
    async def get_response_status(self) -> Dict[str, Any]:
        """Get autonomous response system status"""
        try:
            status = {
                "active": self.active,
                "registered_responses": len(self.orchestrator.response_catalog) if self.orchestrator else 0,
                "active_responses": len(self.orchestrator.active_responses) if self.orchestrator else 0,
                "execution_history_size": len(self.orchestrator.execution_history) if self.orchestrator else 0,
                "response_metrics": dict(self.response_metrics),
                "adaptive_countermeasures": len(self.countermeasures),
                "last_updated": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Graceful shutdown of autonomous response system"""
        self.active = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Autonomous Security Response system shutdown completed")


# Global ASR instance
asr = None

async def get_autonomous_security_response() -> AutonomousSecurityResponse:
    """Get or create ASR instance"""
    global asr
    if not asr:
        asr = AutonomousSecurityResponse(
            redis_url="redis://localhost:6379",
            database_url="postgresql://localhost:5432/nautilus"
        )
        await asr.initialize()
    return asr


async def respond_to_security_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for autonomous security response
    """
    try:
        asr_system = await get_autonomous_security_response()
        executions = await asr_system.process_security_event(event_data)
        
        return {
            "responses_executed": len(executions),
            "execution_details": [
                {
                    "execution_id": ex.execution_id,
                    "response_id": ex.response_id,
                    "status": ex.status,
                    "actions_executed": ex.actions_executed
                }
                for ex in executions
            ]
        }
        
    except Exception as e:
        logger.error(f"Autonomous security response failed: {e}")
        return {"error": str(e)}
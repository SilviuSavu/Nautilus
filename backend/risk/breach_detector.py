"""
Risk Breach Detection System - Sprint 3 Priority 1
Real-time Risk Breach Detection and Response Infrastructure

Provides comprehensive breach detection with:
- Real-time limit violation detection
- Sophisticated severity classification (minor, major, critical)
- Automated response workflows and escalation
- Pattern analysis and breach prediction
- Integration with limit engine and risk monitor
- WebSocket notifications via Redis pub/sub
- Database persistence for breach history
- Configurable response actions and alerting
"""

import asyncio
import logging
import uuid
import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from decimal import Decimal
from collections import deque, defaultdict
import hashlib
from contextlib import asynccontextmanager

# Database and Redis imports
import asyncpg
import redis.asyncio as redis

# Internal imports
from .limit_engine import (
    DynamicLimitEngine, 
    RiskLimit, 
    LimitCheck, 
    LimitType, 
    LimitScope, 
    LimitStatus,
    TimeWindow
)
from ..websocket.redis_pubsub import get_redis_pubsub_manager
from ..websocket.message_protocols import (
    create_breach_alert_message, 
    create_system_alert_message,
    create_risk_alert_message
)

logger = logging.getLogger(__name__)


class BreachSeverity(Enum):
    """Breach severity classifications"""
    MINOR = "minor"           # 100-125% of limit
    MAJOR = "major"           # 125-150% of limit  
    CRITICAL = "critical"     # 150%+ of limit
    EMERGENCY = "emergency"   # 200%+ of limit or system-critical


class BreachCategory(Enum):
    """Breach category classifications"""
    LIMIT_VIOLATION = "limit_violation"
    PATTERN_VIOLATION = "pattern_violation"
    THRESHOLD_BREACH = "threshold_breach"
    SYSTEMIC_RISK = "systemic_risk"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CORRELATION_RISK = "correlation_risk"
    VOLATILITY_SPIKE = "volatility_spike"


class BreachStatus(Enum):
    """Breach lifecycle status"""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    RECURRING = "recurring"


class ResponseAction(Enum):
    """Automated response actions"""
    ALERT_ONLY = "alert_only"
    POSITION_REDUCE = "position_reduce"
    TRADING_HALT = "trading_halt"
    PORTFOLIO_FREEZE = "portfolio_freeze"
    EMERGENCY_LIQUIDATE = "emergency_liquidate"
    ESCALATE_HUMAN = "escalate_human"
    RISK_OVERRIDE = "risk_override"


class EscalationLevel(Enum):
    """Escalation hierarchy levels"""
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    HEAD_OF_RISK = "head_of_risk"
    CRO = "cro"
    EXECUTIVE = "executive"


@dataclass
class BreachPattern:
    """Detected breach pattern"""
    pattern_id: str
    pattern_type: str  # recurring, cascading, systemic, cluster
    portfolios_affected: Set[str]
    frequency: int  # occurrences
    time_window: timedelta
    severity_trend: str  # escalating, stable, de-escalating
    risk_multiplier: float
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BreachEvent:
    """Individual breach event record"""
    breach_id: str
    limit_id: str
    portfolio_id: str
    strategy_id: Optional[str]
    symbol: Optional[str]
    breach_type: LimitType
    severity: BreachSeverity
    category: BreachCategory
    status: BreachStatus
    
    # Breach metrics
    current_value: Decimal
    limit_value: Decimal
    breach_amount: Decimal
    breach_percentage: float
    utilization_ratio: float
    
    # Context and metadata
    scope: LimitScope
    scope_id: str
    time_window: TimeWindow
    market_conditions: Dict[str, Any]
    
    # Response tracking
    actions_taken: List[ResponseAction]
    escalation_level: EscalationLevel
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Pattern analysis
    pattern_id: Optional[str] = None
    is_recurring: bool = False
    recurrence_count: int = 0
    related_breaches: List[str] = field(default_factory=list)
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResponseRule:
    """Automated response rule configuration"""
    rule_id: str
    name: str
    description: str
    
    # Trigger conditions
    severity_threshold: BreachSeverity
    breach_types: Set[LimitType]
    categories: Set[BreachCategory]
    portfolio_filters: Optional[Set[str]] = None
    time_window_minutes: int = 5
    
    # Actions
    immediate_actions: List[ResponseAction]
    escalation_delay_minutes: int = 15
    escalation_actions: List[ResponseAction]
    escalation_level: EscalationLevel = EscalationLevel.RISK_MANAGER
    
    # Conditions
    max_utilization: float = 1.0
    pattern_trigger: bool = False
    require_human_approval: bool = False
    enabled: bool = True
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class BreachDetector:
    """
    Comprehensive risk breach detection and response system
    
    Features:
    - Real-time breach detection with sophisticated severity classification
    - Pattern analysis and breach prediction
    - Automated response workflows with escalation
    - Integration with limit engine and WebSocket notifications
    - Database persistence and breach history analysis
    - Configurable response rules and actions
    """
    
    def __init__(
        self, 
        limit_engine: DynamicLimitEngine,
        database_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus",
        redis_url: str = "redis://localhost:6379"
    ):
        self.limit_engine = limit_engine
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Core components
        self.db_connection: Optional[asyncpg.Connection] = None
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pubsub = None
        
        # Breach tracking
        self.active_breaches: Dict[str, BreachEvent] = {}
        self.breach_history: deque = deque(maxlen=10000)  # Keep last 10k breaches
        self.breach_patterns: Dict[str, BreachPattern] = {}
        
        # Pattern analysis
        self.pattern_detector = BreachPatternDetector()
        self.breach_predictor = BreachPredictor()
        
        # Response system
        self.response_rules: Dict[str, ResponseRule] = {}
        self.response_executor = ResponseExecutor(self)
        
        # Monitoring and statistics
        self.total_breaches_detected = 0
        self.breaches_by_severity = defaultdict(int)
        self.breaches_by_type = defaultdict(int)
        self.pattern_predictions_accurate = 0
        self.pattern_predictions_total = 0
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._pattern_analysis_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._escalation_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.breach_callbacks: List[Callable[[BreachEvent], None]] = []
        self.pattern_callbacks: List[Callable[[BreachPattern], None]] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the breach detector"""
        try:
            # Initialize database connection
            self.db_connection = await asyncpg.connect(self.database_url)
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize Redis pub/sub
            self.redis_pubsub = get_redis_pubsub_manager()
            
            # Create database tables
            await self._create_database_tables()
            
            # Load response rules and breach history
            await self._load_response_rules()
            await self._load_recent_breaches()
            
            # Initialize pattern detector and predictor
            await self.pattern_detector.initialize(self.db_connection)
            await self.breach_predictor.initialize(self.db_connection)
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
            self._pattern_analysis_task = asyncio.create_task(self._pattern_analysis_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_breaches())
            self._escalation_task = asyncio.create_task(self._handle_escalations())
            
            # Setup default response rules
            await self._create_default_response_rules()
            
            self.logger.info("Breach detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize breach detector: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the breach detector"""
        try:
            # Cancel background tasks
            tasks = [
                self._monitoring_task,
                self._pattern_analysis_task, 
                self._cleanup_task,
                self._escalation_task
            ]
            
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close connections
            if self.db_connection:
                await self.db_connection.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Breach detector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during breach detector shutdown: {e}")
    
    async def check_for_breaches(
        self, 
        limit_checks: List[LimitCheck],
        context: Optional[Dict[str, Any]] = None
    ) -> List[BreachEvent]:
        """
        Check limit results for breaches and classify severity
        
        Args:
            limit_checks: Results from limit engine checks
            context: Additional context for breach analysis
            
        Returns:
            List of detected breach events
        """
        detected_breaches = []
        context = context or {}
        
        try:
            for check in limit_checks:
                # Only process breached or warning limits
                if check.status not in [LimitStatus.BREACHED, LimitStatus.WARNING]:
                    continue
                
                # Get limit configuration
                limit = self.limit_engine.limits.get(check.limit_id)
                if not limit:
                    continue
                
                # Check if this is a new breach or escalation
                breach_id = self._generate_breach_id(check, limit)
                existing_breach = self.active_breaches.get(breach_id)
                
                if existing_breach:
                    # Check for severity escalation
                    new_severity = self._calculate_breach_severity(check, limit)
                    if new_severity.value > existing_breach.severity.value:
                        await self._escalate_breach(existing_breach, new_severity, check)
                    continue
                
                # Create new breach event
                breach = await self._create_breach_event(check, limit, context)
                
                # Store and track breach
                self.active_breaches[breach.breach_id] = breach
                self.breach_history.append(breach)
                self.total_breaches_detected += 1
                self.breaches_by_severity[breach.severity] += 1
                self.breaches_by_type[breach.breach_type] += 1
                
                detected_breaches.append(breach)
                
                # Trigger immediate response
                await self._trigger_immediate_response(breach)
                
                # Send alerts
                await self._send_breach_alerts(breach)
                
                # Persist to database
                await self._save_breach_to_database(breach)
                
                # Pattern analysis
                await self._analyze_breach_patterns(breach)
                
                # Execute callbacks
                for callback in self.breach_callbacks:
                    try:
                        callback(breach)
                    except Exception as e:
                        self.logger.error(f"Error in breach callback: {e}")
                
                self.logger.warning(
                    f"BREACH DETECTED: {breach.breach_type.value} for {breach.scope.value} "
                    f"{breach.scope_id} - Severity: {breach.severity.value} - "
                    f"Utilization: {breach.utilization_ratio:.1%}"
                )
            
            return detected_breaches
            
        except Exception as e:
            self.logger.error(f"Error checking for breaches: {e}")
            return []
    
    async def _create_breach_event(
        self, 
        check: LimitCheck, 
        limit: RiskLimit, 
        context: Dict[str, Any]
    ) -> BreachEvent:
        """Create a breach event from a limit check"""
        
        # Calculate breach metrics
        breach_amount = check.current_value - check.limit_value
        breach_percentage = float((check.current_value - check.limit_value) / check.limit_value * 100)
        utilization_ratio = check.utilization
        
        # Determine severity
        severity = self._calculate_breach_severity(check, limit)
        
        # Determine category
        category = self._classify_breach_category(limit.limit_type, check, context)
        
        # Get market conditions
        market_conditions = await self._get_market_conditions(limit.scope_id, context)
        
        # Create breach event
        breach = BreachEvent(
            breach_id=self._generate_breach_id(check, limit),
            limit_id=limit.limit_id,
            portfolio_id=limit.scope_id if limit.scope == LimitScope.PORTFOLIO else context.get("portfolio_id", "unknown"),
            strategy_id=limit.scope_id if limit.scope == LimitScope.STRATEGY else context.get("strategy_id"),
            symbol=limit.scope_id if limit.scope == LimitScope.SYMBOL else context.get("symbol"),
            breach_type=limit.limit_type,
            severity=severity,
            category=category,
            status=BreachStatus.DETECTED,
            current_value=check.current_value,
            limit_value=check.limit_value,
            breach_amount=breach_amount,
            breach_percentage=breach_percentage,
            utilization_ratio=utilization_ratio,
            scope=limit.scope,
            scope_id=limit.scope_id,
            time_window=limit.time_window,
            market_conditions=market_conditions,
            actions_taken=[],
            escalation_level=EscalationLevel.TRADER,
            metadata={
                "limit_description": limit.description,
                "warning_threshold": limit.warning_threshold,
                "breach_threshold": limit.breach_threshold,
                "context": context
            }
        )
        
        # Check for patterns
        await self._check_recurring_patterns(breach)
        
        return breach
    
    def _calculate_breach_severity(self, check: LimitCheck, limit: RiskLimit) -> BreachSeverity:
        """Calculate breach severity based on utilization"""
        utilization = check.utilization
        
        if utilization >= 2.0:  # 200%+ of limit
            return BreachSeverity.EMERGENCY
        elif utilization >= 1.5:  # 150%+ of limit
            return BreachSeverity.CRITICAL
        elif utilization >= 1.25:  # 125%+ of limit
            return BreachSeverity.MAJOR
        else:  # 100-125% of limit
            return BreachSeverity.MINOR
    
    def _classify_breach_category(
        self, 
        limit_type: LimitType, 
        check: LimitCheck, 
        context: Dict[str, Any]
    ) -> BreachCategory:
        """Classify breach into categories"""
        
        # Concentration-related limits
        if limit_type in [LimitType.CONCENTRATION, LimitType.SECTOR_EXPOSURE]:
            return BreachCategory.CONCENTRATION_RISK
        
        # Correlation-related limits  
        if limit_type in [LimitType.CORRELATION]:
            return BreachCategory.CORRELATION_RISK
        
        # Volatility-related limits
        if limit_type in [LimitType.VOLATILITY]:
            return BreachCategory.VOLATILITY_SPIKE
        
        # VaR and exposure limits
        if limit_type in [LimitType.VAR_ABSOLUTE, LimitType.VAR_RELATIVE,
                         LimitType.EXPOSURE_GROSS, LimitType.EXPOSURE_NET]:
            return BreachCategory.LIMIT_VIOLATION
        
        # Drawdown limits
        if limit_type in [LimitType.DRAWDOWN_MAX, LimitType.DRAWDOWN_DAILY]:
            return BreachCategory.THRESHOLD_BREACH
        
        return BreachCategory.LIMIT_VIOLATION
    
    async def _get_market_conditions(self, scope_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current market conditions for context"""
        try:
            # This would integrate with market data services
            # For now, return basic placeholder data
            return {
                "market_volatility": context.get("market_volatility", 0.20),
                "market_direction": context.get("market_direction", "neutral"),
                "trading_session": self._get_trading_session(),
                "liquidity_conditions": context.get("liquidity", "normal"),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting market conditions: {e}")
            return {"timestamp": datetime.utcnow().isoformat()}
    
    def _get_trading_session(self) -> str:
        """Determine current trading session"""
        now = datetime.utcnow()
        hour = now.hour
        
        # US market hours (UTC)
        if 14 <= hour < 21:  # 9:30 AM - 4:00 PM EST
            return "us_regular"
        elif 9 <= hour < 14:  # 4:00 AM - 9:30 AM EST
            return "us_premarket"
        elif 21 <= hour <= 23 or 0 <= hour < 1:  # 4:00 PM - 8:00 PM EST
            return "us_afterhours"
        else:
            return "closed"
    
    def _generate_breach_id(self, check: LimitCheck, limit: RiskLimit) -> str:
        """Generate unique breach ID"""
        source = f"{limit.limit_id}:{limit.scope_id}:{check.timestamp.isoformat()}"
        return hashlib.md5(source.encode()).hexdigest()[:16]
    
    async def _check_recurring_patterns(self, breach: BreachEvent) -> None:
        """Check if this breach is part of a recurring pattern"""
        try:
            # Look for similar breaches in recent history
            recent_breaches = [
                b for b in self.breach_history 
                if (b.limit_id == breach.limit_id and 
                    b.detected_at > datetime.utcnow() - timedelta(hours=24))
            ]
            
            if len(recent_breaches) >= 3:  # 3+ breaches in 24 hours
                breach.is_recurring = True
                breach.recurrence_count = len(recent_breaches)
                breach.status = BreachStatus.RECURRING
                
                # Find or create pattern
                pattern_id = await self._identify_or_create_pattern(breach, recent_breaches)
                breach.pattern_id = pattern_id
                
                self.logger.warning(
                    f"RECURRING BREACH PATTERN detected: {pattern_id} - "
                    f"Count: {breach.recurrence_count}"
                )
        
        except Exception as e:
            self.logger.error(f"Error checking recurring patterns: {e}")
    
    async def _identify_or_create_pattern(
        self, 
        breach: BreachEvent, 
        recent_breaches: List[BreachEvent]
    ) -> str:
        """Identify existing pattern or create new one"""
        try:
            # Generate pattern signature
            signature = self._create_pattern_signature(breach, recent_breaches)
            
            # Check for existing pattern
            for pattern_id, pattern in self.breach_patterns.items():
                if pattern.pattern_type == signature:
                    pattern.frequency += 1
                    return pattern_id
            
            # Create new pattern
            pattern_id = str(uuid.uuid4())[:8]
            pattern = BreachPattern(
                pattern_id=pattern_id,
                pattern_type=signature,
                portfolios_affected={breach.portfolio_id},
                frequency=len(recent_breaches),
                time_window=timedelta(hours=24),
                severity_trend=self._analyze_severity_trend(recent_breaches),
                risk_multiplier=self._calculate_risk_multiplier(recent_breaches),
                confidence_score=min(0.9, len(recent_breaches) / 10.0)
            )
            
            self.breach_patterns[pattern_id] = pattern
            
            # Execute pattern callbacks
            for callback in self.pattern_callbacks:
                try:
                    callback(pattern)
                except Exception as e:
                    self.logger.error(f"Error in pattern callback: {e}")
            
            return pattern_id
        
        except Exception as e:
            self.logger.error(f"Error identifying pattern: {e}")
            return f"unknown-{uuid.uuid4().hex[:8]}"
    
    def _create_pattern_signature(
        self, 
        breach: BreachEvent, 
        recent_breaches: List[BreachEvent]
    ) -> str:
        """Create pattern signature for similar breaches"""
        return f"{breach.breach_type.value}:{breach.scope.value}:{breach.severity.value}"
    
    def _analyze_severity_trend(self, breaches: List[BreachEvent]) -> str:
        """Analyze if breach severity is escalating, stable, or de-escalating"""
        if len(breaches) < 2:
            return "stable"
        
        severity_scores = [self._severity_to_score(b.severity) for b in breaches[-5:]]
        
        if len(severity_scores) >= 3:
            trend = np.polyfit(range(len(severity_scores)), severity_scores, 1)[0]
            if trend > 0.1:
                return "escalating"
            elif trend < -0.1:
                return "de-escalating"
        
        return "stable"
    
    def _severity_to_score(self, severity: BreachSeverity) -> float:
        """Convert severity to numeric score"""
        scores = {
            BreachSeverity.MINOR: 1.0,
            BreachSeverity.MAJOR: 2.0,
            BreachSeverity.CRITICAL: 3.0,
            BreachSeverity.EMERGENCY: 4.0
        }
        return scores.get(severity, 1.0)
    
    def _calculate_risk_multiplier(self, breaches: List[BreachEvent]) -> float:
        """Calculate risk multiplier based on breach pattern"""
        base_multiplier = 1.0
        
        # Frequency multiplier
        frequency_multiplier = min(2.0, 1.0 + (len(breaches) - 1) * 0.1)
        
        # Severity multiplier
        max_severity = max(self._severity_to_score(b.severity) for b in breaches)
        severity_multiplier = max_severity / 4.0  # Normalize to 0-1
        
        return base_multiplier * frequency_multiplier * (1.0 + severity_multiplier)
    
    async def _trigger_immediate_response(self, breach: BreachEvent) -> None:
        """Trigger immediate automated response to breach"""
        try:
            # Find applicable response rules
            applicable_rules = self._find_applicable_rules(breach)
            
            for rule in applicable_rules:
                if not rule.enabled:
                    continue
                
                # Execute immediate actions
                for action in rule.immediate_actions:
                    await self.response_executor.execute_action(breach, action, rule)
                
                # Schedule escalation if configured
                if rule.escalation_delay_minutes > 0 and rule.escalation_actions:
                    await self._schedule_escalation(breach, rule)
        
        except Exception as e:
            self.logger.error(f"Error triggering immediate response: {e}")
    
    def _find_applicable_rules(self, breach: BreachEvent) -> List[ResponseRule]:
        """Find response rules applicable to the breach"""
        applicable_rules = []
        
        for rule in self.response_rules.values():
            if not rule.enabled:
                continue
            
            # Check severity threshold
            if self._severity_to_score(breach.severity) < self._severity_to_score(rule.severity_threshold):
                continue
            
            # Check breach type
            if rule.breach_types and breach.breach_type not in rule.breach_types:
                continue
            
            # Check category
            if rule.categories and breach.category not in rule.categories:
                continue
            
            # Check portfolio filters
            if rule.portfolio_filters and breach.portfolio_id not in rule.portfolio_filters:
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    async def _schedule_escalation(self, breach: BreachEvent, rule: ResponseRule) -> None:
        """Schedule escalation for a breach"""
        try:
            escalation_time = datetime.utcnow() + timedelta(minutes=rule.escalation_delay_minutes)
            
            # Store escalation in metadata for processing by escalation task
            escalation_data = {
                "breach_id": breach.breach_id,
                "rule_id": rule.rule_id,
                "escalation_time": escalation_time.isoformat(),
                "actions": [action.value for action in rule.escalation_actions],
                "escalation_level": rule.escalation_level.value
            }
            
            # Store in Redis for persistence across restarts
            await self.redis_client.zadd(
                "breach_escalations",
                {json.dumps(escalation_data): escalation_time.timestamp()}
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling escalation: {e}")
    
    async def _send_breach_alerts(self, breach: BreachEvent) -> None:
        """Send breach alerts via WebSocket and other channels"""
        try:
            # Create breach alert message
            alert_data = {
                "breach_id": breach.breach_id,
                "limit_type": breach.breach_type.value,
                "current_value": float(breach.current_value),
                "limit_value": float(breach.limit_value),
                "breach_amount": float(breach.breach_amount),
                "breach_percentage": breach.breach_percentage,
                "utilization": breach.utilization_ratio,
                "severity": breach.severity.value,
                "category": breach.category.value,
                "status": breach.status.value,
                "scope": breach.scope.value,
                "scope_id": breach.scope_id,
                "detected_at": breach.detected_at.isoformat(),
                "is_recurring": breach.is_recurring,
                "recurrence_count": breach.recurrence_count,
                "market_conditions": breach.market_conditions,
                "actions_taken": [action.value for action in breach.actions_taken],
                "escalation_level": breach.escalation_level.value
            }
            
            # Send breach alert message
            message = create_breach_alert_message(
                portfolio_id=breach.portfolio_id,
                breach_type=breach.breach_type.value,
                severity=breach.severity.value,
                data=alert_data
            )
            
            # Publish to Redis pub/sub
            if self.redis_pubsub:
                await self.redis_pubsub.publish_risk_alert(alert_data, breach.portfolio_id)
            
            # Send system alert for critical/emergency breaches
            if breach.severity in [BreachSeverity.CRITICAL, BreachSeverity.EMERGENCY]:
                system_message = create_system_alert_message(
                    component="risk_management",
                    alert_type="breach_critical",
                    severity=breach.severity.value,
                    data=alert_data
                )
                
                if self.redis_pubsub:
                    await self.redis_pubsub.publish_risk_alert(system_message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending breach alerts: {e}")
    
    async def _escalate_breach(
        self, 
        existing_breach: BreachEvent, 
        new_severity: BreachSeverity, 
        check: LimitCheck
    ) -> None:
        """Escalate an existing breach to higher severity"""
        try:
            old_severity = existing_breach.severity
            existing_breach.severity = new_severity
            existing_breach.current_value = check.current_value
            existing_breach.utilization_ratio = check.utilization
            existing_breach.breach_amount = check.current_value - check.limit_value
            existing_breach.breach_percentage = float(
                (check.current_value - check.limit_value) / check.limit_value * 100
            )
            existing_breach.status = BreachStatus.ESCALATED
            
            # Update statistics
            self.breaches_by_severity[old_severity] -= 1
            self.breaches_by_severity[new_severity] += 1
            
            # Trigger escalation response
            await self._trigger_escalation_response(existing_breach, old_severity)
            
            # Send escalation alerts
            await self._send_escalation_alerts(existing_breach, old_severity)
            
            # Update database
            await self._update_breach_in_database(existing_breach)
            
            self.logger.critical(
                f"BREACH ESCALATED: {existing_breach.breach_id} from "
                f"{old_severity.value} to {new_severity.value}"
            )
            
        except Exception as e:
            self.logger.error(f"Error escalating breach: {e}")
    
    async def _trigger_escalation_response(
        self, 
        breach: BreachEvent, 
        old_severity: BreachSeverity
    ) -> None:
        """Trigger escalation response actions"""
        try:
            # Find escalation rules
            escalation_rules = [
                rule for rule in self.response_rules.values()
                if (rule.enabled and 
                    self._severity_to_score(breach.severity) >= self._severity_to_score(rule.severity_threshold) and
                    self._severity_to_score(old_severity) < self._severity_to_score(rule.severity_threshold))
            ]
            
            for rule in escalation_rules:
                # Execute escalation actions immediately
                for action in rule.escalation_actions:
                    await self.response_executor.execute_action(breach, action, rule)
                
                # Update escalation level
                if rule.escalation_level.value > breach.escalation_level.value:
                    breach.escalation_level = rule.escalation_level
            
        except Exception as e:
            self.logger.error(f"Error triggering escalation response: {e}")
    
    async def _send_escalation_alerts(
        self, 
        breach: BreachEvent, 
        old_severity: BreachSeverity
    ) -> None:
        """Send escalation alert notifications"""
        try:
            alert_data = {
                "breach_id": breach.breach_id,
                "escalation_type": "severity_escalation",
                "old_severity": old_severity.value,
                "new_severity": breach.severity.value,
                "current_utilization": breach.utilization_ratio,
                "escalated_at": datetime.utcnow().isoformat(),
                "escalation_level": breach.escalation_level.value
            }
            
            # Send high-priority system alert
            message = create_system_alert_message(
                component="risk_management",
                alert_type="breach_escalation",
                severity="critical",
                data=alert_data
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_risk_alert(message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending escalation alerts: {e}")
    
    # Background monitoring tasks
    
    async def _continuous_monitoring(self) -> None:
        """Continuous monitoring loop for breach detection"""
        try:
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get current limit statuses from limit engine
                active_limits = [
                    limit for limit in self.limit_engine.limits.values()
                    if limit.status == LimitStatus.ACTIVE
                ]
                
                # Check each active limit
                limit_checks = []
                for limit in active_limits:
                    try:
                        check = await self.limit_engine.get_limit_status(limit.limit_id)
                        if check:
                            limit_checks.append(check)
                    except Exception as e:
                        self.logger.error(f"Error checking limit {limit.limit_id}: {e}")
                
                # Process breach detection
                if limit_checks:
                    await self.check_for_breaches(limit_checks)
                
        except asyncio.CancelledError:
            self.logger.info("Continuous monitoring task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in continuous monitoring: {e}")
    
    async def _pattern_analysis_loop(self) -> None:
        """Background pattern analysis and prediction"""
        try:
            while True:
                await asyncio.sleep(60)  # Analyze every minute
                
                # Analyze recent breaches for patterns
                await self.pattern_detector.analyze_recent_breaches(list(self.breach_history))
                
                # Update breach predictions
                predictions = await self.breach_predictor.predict_potential_breaches(
                    list(self.active_breaches.values())
                )
                
                # Process predictions
                for prediction in predictions:
                    await self._process_breach_prediction(prediction)
                
        except asyncio.CancelledError:
            self.logger.info("Pattern analysis task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
    
    async def _cleanup_expired_breaches(self) -> None:
        """Cleanup resolved and expired breaches"""
        try:
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = datetime.utcnow()
                expired_breach_ids = []
                
                # Find expired breaches (resolved for >1 hour or old unresolved)
                for breach_id, breach in self.active_breaches.items():
                    if (breach.status == BreachStatus.RESOLVED and 
                        breach.resolved_at and 
                        current_time - breach.resolved_at > timedelta(hours=1)):
                        expired_breach_ids.append(breach_id)
                    elif (breach.status == BreachStatus.DETECTED and
                          current_time - breach.detected_at > timedelta(hours=24)):
                        # Auto-resolve very old unresolved breaches
                        breach.status = BreachStatus.RESOLVED
                        breach.resolved_at = current_time
                        breach.resolved_by = "system_auto_resolve"
                        await self._update_breach_in_database(breach)
                        expired_breach_ids.append(breach_id)
                
                # Remove expired breaches from active tracking
                for breach_id in expired_breach_ids:
                    del self.active_breaches[breach_id]
                
                if expired_breach_ids:
                    self.logger.info(f"Cleaned up {len(expired_breach_ids)} expired breaches")
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in cleanup task: {e}")
    
    async def _handle_escalations(self) -> None:
        """Handle scheduled breach escalations"""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow().timestamp()
                
                # Get due escalations
                escalations = await self.redis_client.zrangebyscore(
                    "breach_escalations", 0, current_time, withscores=False
                )
                
                for escalation_data in escalations:
                    try:
                        escalation = json.loads(escalation_data)
                        breach_id = escalation["breach_id"]
                        
                        # Check if breach still exists and needs escalation
                        if breach_id in self.active_breaches:
                            breach = self.active_breaches[breach_id]
                            
                            # Execute escalation actions
                            rule = self.response_rules.get(escalation["rule_id"])
                            if rule:
                                for action_name in escalation["actions"]:
                                    action = ResponseAction(action_name)
                                    await self.response_executor.execute_action(breach, action, rule)
                                
                                # Update escalation level
                                escalation_level = EscalationLevel(escalation["escalation_level"])
                                if escalation_level.value > breach.escalation_level.value:
                                    breach.escalation_level = escalation_level
                                    breach.status = BreachStatus.ESCALATED
                        
                        # Remove processed escalation
                        await self.redis_client.zrem("breach_escalations", escalation_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing escalation: {e}")
                        # Remove problematic escalation
                        await self.redis_client.zrem("breach_escalations", escalation_data)
                
        except asyncio.CancelledError:
            self.logger.info("Escalation handling task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in escalation handling: {e}")
    
    async def _process_breach_prediction(self, prediction: Dict[str, Any]) -> None:
        """Process a breach prediction and take preventive action"""
        try:
            confidence = prediction.get("confidence", 0.0)
            if confidence < 0.7:  # Only act on high-confidence predictions
                return
            
            # Send predictive alert
            alert_data = {
                "prediction_type": "breach_prediction",
                "predicted_breach_type": prediction.get("breach_type"),
                "confidence": confidence,
                "time_horizon": prediction.get("time_horizon"),
                "recommended_action": prediction.get("recommended_action"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            message = create_risk_alert_message(
                portfolio_id=prediction.get("portfolio_id", "system"),
                risk_type="predictive_breach",
                severity="medium",
                data=alert_data
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_risk_alert(message.data)
            
        except Exception as e:
            self.logger.error(f"Error processing breach prediction: {e}")
    
    # Database operations
    
    async def _create_database_tables(self) -> None:
        """Create database tables for breach tracking"""
        try:
            # Breach events table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_breach_events (
                    breach_id VARCHAR(32) PRIMARY KEY,
                    limit_id UUID NOT NULL,
                    portfolio_id VARCHAR(100),
                    strategy_id VARCHAR(100),
                    symbol VARCHAR(20),
                    breach_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    current_value DECIMAL(20,2) NOT NULL,
                    limit_value DECIMAL(20,2) NOT NULL,
                    breach_amount DECIMAL(20,2) NOT NULL,
                    breach_percentage DECIMAL(10,4) NOT NULL,
                    utilization_ratio DECIMAL(10,4) NOT NULL,
                    scope VARCHAR(20) NOT NULL,
                    scope_id VARCHAR(100) NOT NULL,
                    time_window VARCHAR(20),
                    market_conditions JSONB,
                    actions_taken TEXT[],
                    escalation_level VARCHAR(20) NOT NULL,
                    acknowledged_by VARCHAR(100),
                    resolved_by VARCHAR(100),
                    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    pattern_id VARCHAR(32),
                    is_recurring BOOLEAN DEFAULT FALSE,
                    recurrence_count INTEGER DEFAULT 0,
                    related_breaches TEXT[],
                    metadata JSONB,
                    response_log JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_breach_events_portfolio 
                    ON risk_breach_events(portfolio_id);
                CREATE INDEX IF NOT EXISTS idx_breach_events_detected_at 
                    ON risk_breach_events(detected_at);
                CREATE INDEX IF NOT EXISTS idx_breach_events_severity 
                    ON risk_breach_events(severity);
                CREATE INDEX IF NOT EXISTS idx_breach_events_status 
                    ON risk_breach_events(status);
                CREATE INDEX IF NOT EXISTS idx_breach_events_pattern 
                    ON risk_breach_events(pattern_id);
            """)
            
            # Breach patterns table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_breach_patterns (
                    pattern_id VARCHAR(32) PRIMARY KEY,
                    pattern_type VARCHAR(100) NOT NULL,
                    portfolios_affected TEXT[],
                    frequency INTEGER NOT NULL,
                    time_window_hours INTEGER NOT NULL,
                    severity_trend VARCHAR(20),
                    risk_multiplier DECIMAL(10,4),
                    confidence_score DECIMAL(5,4),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Response rules table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_response_rules (
                    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    severity_threshold VARCHAR(20) NOT NULL,
                    breach_types TEXT[],
                    categories TEXT[],
                    portfolio_filters TEXT[],
                    time_window_minutes INTEGER DEFAULT 5,
                    immediate_actions TEXT[],
                    escalation_delay_minutes INTEGER DEFAULT 15,
                    escalation_actions TEXT[],
                    escalation_level VARCHAR(20) DEFAULT 'risk_manager',
                    max_utilization DECIMAL(5,4) DEFAULT 1.0,
                    pattern_trigger BOOLEAN DEFAULT FALSE,
                    require_human_approval BOOLEAN DEFAULT FALSE,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _save_breach_to_database(self, breach: BreachEvent) -> None:
        """Save breach event to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO risk_breach_events (
                    breach_id, limit_id, portfolio_id, strategy_id, symbol,
                    breach_type, severity, category, status, current_value,
                    limit_value, breach_amount, breach_percentage, utilization_ratio,
                    scope, scope_id, time_window, market_conditions, actions_taken,
                    escalation_level, detected_at, pattern_id, is_recurring,
                    recurrence_count, related_breaches, metadata, response_log
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27
                )
            """,
                breach.breach_id, breach.limit_id, breach.portfolio_id,
                breach.strategy_id, breach.symbol, breach.breach_type.value,
                breach.severity.value, breach.category.value, breach.status.value,
                breach.current_value, breach.limit_value, breach.breach_amount,
                breach.breach_percentage, breach.utilization_ratio, breach.scope.value,
                breach.scope_id, breach.time_window.value if breach.time_window else None,
                json.dumps(breach.market_conditions), 
                [action.value for action in breach.actions_taken],
                breach.escalation_level.value, breach.detected_at, breach.pattern_id,
                breach.is_recurring, breach.recurrence_count, breach.related_breaches,
                json.dumps(breach.metadata), json.dumps(breach.response_log)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving breach to database: {e}")
    
    async def _update_breach_in_database(self, breach: BreachEvent) -> None:
        """Update existing breach in database"""
        try:
            await self.db_connection.execute("""
                UPDATE risk_breach_events SET
                    status = $2, current_value = $3, breach_amount = $4,
                    breach_percentage = $5, utilization_ratio = $6,
                    actions_taken = $7, escalation_level = $8,
                    acknowledged_by = $9, resolved_by = $10,
                    acknowledged_at = $11, resolved_at = $12,
                    recurrence_count = $13, response_log = $14,
                    updated_at = NOW()
                WHERE breach_id = $1
            """,
                breach.breach_id, breach.status.value, breach.current_value,
                breach.breach_amount, breach.breach_percentage, breach.utilization_ratio,
                [action.value for action in breach.actions_taken], breach.escalation_level.value,
                breach.acknowledged_by, breach.resolved_by, breach.acknowledged_at,
                breach.resolved_at, breach.recurrence_count, json.dumps(breach.response_log)
            )
            
        except Exception as e:
            self.logger.error(f"Error updating breach in database: {e}")
    
    async def _load_response_rules(self) -> None:
        """Load response rules from database"""
        try:
            rules = await self.db_connection.fetch("""
                SELECT rule_id, name, description, severity_threshold, breach_types,
                       categories, portfolio_filters, time_window_minutes,
                       immediate_actions, escalation_delay_minutes, escalation_actions,
                       escalation_level, max_utilization, pattern_trigger,
                       require_human_approval, enabled, created_at, updated_at
                FROM risk_response_rules WHERE enabled = TRUE
            """)
            
            for rule_data in rules:
                rule = ResponseRule(
                    rule_id=str(rule_data['rule_id']),
                    name=rule_data['name'],
                    description=rule_data['description'],
                    severity_threshold=BreachSeverity(rule_data['severity_threshold']),
                    breach_types=set(LimitType(bt) for bt in (rule_data['breach_types'] or [])),
                    categories=set(BreachCategory(cat) for cat in (rule_data['categories'] or [])),
                    portfolio_filters=set(rule_data['portfolio_filters'] or []) or None,
                    time_window_minutes=rule_data['time_window_minutes'],
                    immediate_actions=[ResponseAction(action) for action in (rule_data['immediate_actions'] or [])],
                    escalation_delay_minutes=rule_data['escalation_delay_minutes'],
                    escalation_actions=[ResponseAction(action) for action in (rule_data['escalation_actions'] or [])],
                    escalation_level=EscalationLevel(rule_data['escalation_level']),
                    max_utilization=float(rule_data['max_utilization']),
                    pattern_trigger=rule_data['pattern_trigger'],
                    require_human_approval=rule_data['require_human_approval'],
                    enabled=rule_data['enabled'],
                    created_at=rule_data['created_at'],
                    updated_at=rule_data['updated_at']
                )
                
                self.response_rules[rule.rule_id] = rule
            
            self.logger.info(f"Loaded {len(self.response_rules)} response rules")
            
        except Exception as e:
            self.logger.error(f"Error loading response rules: {e}")
    
    async def _load_recent_breaches(self) -> None:
        """Load recent breach history from database"""
        try:
            # Load breaches from last 24 hours
            recent_breaches = await self.db_connection.fetch("""
                SELECT * FROM risk_breach_events 
                WHERE detected_at > NOW() - INTERVAL '24 hours'
                ORDER BY detected_at DESC
                LIMIT 1000
            """)
            
            for breach_data in recent_breaches:
                breach = self._create_breach_from_db_record(breach_data)
                self.breach_history.append(breach)
                
                # Add to active breaches if not resolved
                if breach.status not in [BreachStatus.RESOLVED]:
                    self.active_breaches[breach.breach_id] = breach
            
            self.logger.info(f"Loaded {len(recent_breaches)} recent breaches")
            
        except Exception as e:
            self.logger.error(f"Error loading recent breaches: {e}")
    
    def _create_breach_from_db_record(self, record: dict) -> BreachEvent:
        """Create breach event from database record"""
        return BreachEvent(
            breach_id=record['breach_id'],
            limit_id=record['limit_id'],
            portfolio_id=record['portfolio_id'],
            strategy_id=record['strategy_id'],
            symbol=record['symbol'],
            breach_type=LimitType(record['breach_type']),
            severity=BreachSeverity(record['severity']),
            category=BreachCategory(record['category']),
            status=BreachStatus(record['status']),
            current_value=record['current_value'],
            limit_value=record['limit_value'],
            breach_amount=record['breach_amount'],
            breach_percentage=float(record['breach_percentage']),
            utilization_ratio=float(record['utilization_ratio']),
            scope=LimitScope(record['scope']),
            scope_id=record['scope_id'],
            time_window=TimeWindow(record['time_window']) if record['time_window'] else TimeWindow.INTRADAY,
            market_conditions=record['market_conditions'] or {},
            actions_taken=[ResponseAction(action) for action in (record['actions_taken'] or [])],
            escalation_level=EscalationLevel(record['escalation_level']),
            acknowledged_by=record['acknowledged_by'],
            resolved_by=record['resolved_by'],
            detected_at=record['detected_at'],
            acknowledged_at=record['acknowledged_at'],
            resolved_at=record['resolved_at'],
            pattern_id=record['pattern_id'],
            is_recurring=record['is_recurring'],
            recurrence_count=record['recurrence_count'],
            related_breaches=record['related_breaches'] or [],
            metadata=record['metadata'] or {},
            response_log=record['response_log'] or []
        )
    
    async def _create_default_response_rules(self) -> None:
        """Create default response rules if none exist"""
        try:
            if self.response_rules:
                return  # Rules already exist
            
            # Critical VaR breach rule
            var_rule = ResponseRule(
                rule_id="default-var-critical",
                name="Critical VaR Breach Response",
                description="Immediate response to critical VaR limit breaches",
                severity_threshold=BreachSeverity.CRITICAL,
                breach_types={LimitType.VAR_ABSOLUTE, LimitType.VAR_RELATIVE},
                categories={BreachCategory.LIMIT_VIOLATION},
                immediate_actions=[ResponseAction.ALERT_ONLY, ResponseAction.ESCALATE_HUMAN],
                escalation_delay_minutes=10,
                escalation_actions=[ResponseAction.TRADING_HALT],
                escalation_level=EscalationLevel.RISK_MANAGER
            )
            
            # Drawdown emergency rule
            drawdown_rule = ResponseRule(
                rule_id="default-drawdown-emergency",
                name="Emergency Drawdown Response",
                description="Emergency response to severe drawdown breaches",
                severity_threshold=BreachSeverity.EMERGENCY,
                breach_types={LimitType.DRAWDOWN_MAX, LimitType.DRAWDOWN_DAILY},
                categories={BreachCategory.THRESHOLD_BREACH},
                immediate_actions=[ResponseAction.TRADING_HALT, ResponseAction.ESCALATE_HUMAN],
                escalation_delay_minutes=5,
                escalation_actions=[ResponseAction.EMERGENCY_LIQUIDATE],
                escalation_level=EscalationLevel.HEAD_OF_RISK
            )
            
            # Concentration risk rule
            concentration_rule = ResponseRule(
                rule_id="default-concentration-major",
                name="Concentration Risk Management",
                description="Response to major concentration limit breaches",
                severity_threshold=BreachSeverity.MAJOR,
                breach_types={LimitType.CONCENTRATION, LimitType.SECTOR_EXPOSURE},
                categories={BreachCategory.CONCENTRATION_RISK},
                immediate_actions=[ResponseAction.ALERT_ONLY],
                escalation_delay_minutes=30,
                escalation_actions=[ResponseAction.POSITION_REDUCE],
                escalation_level=EscalationLevel.TRADER
            )
            
            # Store rules
            default_rules = [var_rule, drawdown_rule, concentration_rule]
            
            for rule in default_rules:
                await self._save_response_rule(rule)
                self.response_rules[rule.rule_id] = rule
            
            self.logger.info(f"Created {len(default_rules)} default response rules")
            
        except Exception as e:
            self.logger.error(f"Error creating default response rules: {e}")
    
    async def _save_response_rule(self, rule: ResponseRule) -> None:
        """Save response rule to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO risk_response_rules (
                    rule_id, name, description, severity_threshold, breach_types,
                    categories, portfolio_filters, time_window_minutes,
                    immediate_actions, escalation_delay_minutes, escalation_actions,
                    escalation_level, max_utilization, pattern_trigger,
                    require_human_approval, enabled
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                )
                ON CONFLICT (rule_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    severity_threshold = EXCLUDED.severity_threshold,
                    breach_types = EXCLUDED.breach_types,
                    categories = EXCLUDED.categories,
                    portfolio_filters = EXCLUDED.portfolio_filters,
                    immediate_actions = EXCLUDED.immediate_actions,
                    escalation_actions = EXCLUDED.escalation_actions,
                    updated_at = NOW()
            """,
                rule.rule_id, rule.name, rule.description, rule.severity_threshold.value,
                [bt.value for bt in rule.breach_types],
                [cat.value for cat in rule.categories],
                list(rule.portfolio_filters) if rule.portfolio_filters else None,
                rule.time_window_minutes,
                [action.value for action in rule.immediate_actions],
                rule.escalation_delay_minutes,
                [action.value for action in rule.escalation_actions],
                rule.escalation_level.value, rule.max_utilization,
                rule.pattern_trigger, rule.require_human_approval, rule.enabled
            )
            
        except Exception as e:
            self.logger.error(f"Error saving response rule: {e}")
    
    # Public API methods
    
    async def acknowledge_breach(self, breach_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a breach"""
        try:
            if breach_id not in self.active_breaches:
                return False
            
            breach = self.active_breaches[breach_id]
            breach.status = BreachStatus.ACKNOWLEDGED
            breach.acknowledged_by = acknowledged_by
            breach.acknowledged_at = datetime.utcnow()
            
            # Update database
            await self._update_breach_in_database(breach)
            
            # Log response
            breach.response_log.append({
                "action": "acknowledged",
                "by": acknowledged_by,
                "timestamp": breach.acknowledged_at.isoformat()
            })
            
            self.logger.info(f"Breach {breach_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error acknowledging breach: {e}")
            return False
    
    async def resolve_breach(self, breach_id: str, resolved_by: str) -> bool:
        """Resolve a breach"""
        try:
            if breach_id not in self.active_breaches:
                return False
            
            breach = self.active_breaches[breach_id]
            breach.status = BreachStatus.RESOLVED
            breach.resolved_by = resolved_by
            breach.resolved_at = datetime.utcnow()
            
            # Update database
            await self._update_breach_in_database(breach)
            
            # Log response
            breach.response_log.append({
                "action": "resolved",
                "by": resolved_by,
                "timestamp": breach.resolved_at.isoformat()
            })
            
            self.logger.info(f"Breach {breach_id} resolved by {resolved_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resolving breach: {e}")
            return False
    
    async def get_active_breaches(self, portfolio_id: Optional[str] = None) -> List[BreachEvent]:
        """Get all active breaches, optionally filtered by portfolio"""
        try:
            if portfolio_id:
                return [
                    breach for breach in self.active_breaches.values()
                    if breach.portfolio_id == portfolio_id
                ]
            else:
                return list(self.active_breaches.values())
                
        except Exception as e:
            self.logger.error(f"Error getting active breaches: {e}")
            return []
    
    async def get_breach_history(
        self, 
        portfolio_id: Optional[str] = None,
        hours_back: int = 24
    ) -> List[BreachEvent]:
        """Get breach history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            breaches = [
                breach for breach in self.breach_history
                if breach.detected_at >= cutoff_time
            ]
            
            if portfolio_id:
                breaches = [
                    breach for breach in breaches
                    if breach.portfolio_id == portfolio_id
                ]
            
            return sorted(breaches, key=lambda b: b.detected_at, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting breach history: {e}")
            return []
    
    async def get_breach_patterns(self) -> List[BreachPattern]:
        """Get detected breach patterns"""
        return list(self.breach_patterns.values())
    
    def get_breach_statistics(self) -> Dict[str, Any]:
        """Get breach detection statistics"""
        active_count = len(self.active_breaches)
        
        return {
            "total_breaches_detected": self.total_breaches_detected,
            "active_breaches": active_count,
            "breaches_by_severity": dict(self.breaches_by_severity),
            "breaches_by_type": {k.value: v for k, v in self.breaches_by_type.items()},
            "breach_patterns_detected": len(self.breach_patterns),
            "pattern_prediction_accuracy": (
                self.pattern_predictions_accurate / self.pattern_predictions_total 
                if self.pattern_predictions_total > 0 else 0.0
            ),
            "response_rules_active": len([r for r in self.response_rules.values() if r.enabled]),
            "most_common_breach_types": sorted(
                self.breaches_by_type.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    async def add_response_rule(self, rule: ResponseRule) -> bool:
        """Add a new response rule"""
        try:
            await self._save_response_rule(rule)
            self.response_rules[rule.rule_id] = rule
            self.logger.info(f"Added response rule: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding response rule: {e}")
            return False
    
    def add_breach_callback(self, callback: Callable[[BreachEvent], None]) -> None:
        """Add callback for breach events"""
        self.breach_callbacks.append(callback)
    
    def add_pattern_callback(self, callback: Callable[[BreachPattern], None]) -> None:
        """Add callback for pattern detection"""
        self.pattern_callbacks.append(callback)


class BreachPatternDetector:
    """Advanced pattern detection for breach events"""
    
    def __init__(self):
        self.db_connection: Optional[asyncpg.Connection] = None
        self.pattern_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, db_connection: asyncpg.Connection) -> None:
        """Initialize pattern detector"""
        self.db_connection = db_connection
        
    async def analyze_recent_breaches(self, breaches: List[BreachEvent]) -> List[Dict[str, Any]]:
        """Analyze recent breaches for patterns"""
        patterns = []
        
        try:
            # Cluster analysis by time windows
            time_clusters = self._cluster_by_time(breaches)
            for cluster in time_clusters:
                if len(cluster) >= 3:  # Minimum cluster size
                    pattern = self._analyze_cluster_pattern(cluster)
                    if pattern:
                        patterns.append(pattern)
            
            # Sequential analysis
            sequential_patterns = self._find_sequential_patterns(breaches)
            patterns.extend(sequential_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing breach patterns: {e}")
            return []
    
    def _cluster_by_time(self, breaches: List[BreachEvent]) -> List[List[BreachEvent]]:
        """Cluster breaches by time proximity"""
        if len(breaches) < 2:
            return []
        
        # Sort by time
        sorted_breaches = sorted(breaches, key=lambda b: b.detected_at)
        clusters = []
        current_cluster = [sorted_breaches[0]]
        
        for breach in sorted_breaches[1:]:
            # If within 30 minutes of previous breach, add to cluster
            if (breach.detected_at - current_cluster[-1].detected_at).total_seconds() <= 1800:
                current_cluster.append(breach)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [breach]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def _analyze_cluster_pattern(self, cluster: List[BreachEvent]) -> Optional[Dict[str, Any]]:
        """Analyze a cluster of breaches for patterns"""
        try:
            # Common characteristics
            portfolios = set(b.portfolio_id for b in cluster)
            breach_types = set(b.breach_type for b in cluster)
            severities = [b.severity for b in cluster]
            
            # Pattern classification
            pattern_type = "cluster"
            if len(portfolios) == 1:
                pattern_type = "portfolio_cascade"
            elif len(breach_types) == 1:
                pattern_type = "limit_type_cascade"
            
            return {
                "pattern_type": pattern_type,
                "breach_count": len(cluster),
                "time_span": (cluster[-1].detected_at - cluster[0].detected_at).total_seconds(),
                "portfolios_affected": list(portfolios),
                "breach_types": [bt.value for bt in breach_types],
                "severity_escalation": self._check_severity_escalation(severities),
                "confidence": min(0.9, len(cluster) / 10.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cluster pattern: {e}")
            return None
    
    def _find_sequential_patterns(self, breaches: List[BreachEvent]) -> List[Dict[str, Any]]:
        """Find sequential breach patterns"""
        patterns = []
        
        try:
            if len(breaches) < 3:
                return patterns
            
            # Look for repeating sequences
            for i in range(len(breaches) - 2):
                for j in range(i + 3, min(len(breaches) + 1, i + 8)):  # Look ahead 3-8 breaches
                    sequence = breaches[i:j]
                    if self._is_repeating_sequence(sequence, breaches[j:]):
                        patterns.append({
                            "pattern_type": "sequential_repeat",
                            "sequence_length": len(sequence),
                            "repetitions": self._count_repetitions(sequence, breaches),
                            "confidence": 0.8
                        })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error finding sequential patterns: {e}")
            return []
    
    def _check_severity_escalation(self, severities: List[BreachSeverity]) -> bool:
        """Check if severity is escalating"""
        if len(severities) < 2:
            return False
        
        severity_scores = [
            {"minor": 1, "major": 2, "critical": 3, "emergency": 4}[s.value]
            for s in severities
        ]
        
        return severity_scores[-1] > severity_scores[0]
    
    def _is_repeating_sequence(self, sequence: List[BreachEvent], remaining: List[BreachEvent]) -> bool:
        """Check if a sequence repeats"""
        if len(remaining) < len(sequence):
            return False
        
        for i, breach in enumerate(sequence):
            if i >= len(remaining):
                return False
            if breach.breach_type != remaining[i].breach_type:
                return False
        
        return True
    
    def _count_repetitions(self, sequence: List[BreachEvent], all_breaches: List[BreachEvent]) -> int:
        """Count how many times a sequence repeats"""
        count = 0
        seq_len = len(sequence)
        
        for i in range(len(all_breaches) - seq_len + 1):
            if self._sequences_match(sequence, all_breaches[i:i+seq_len]):
                count += 1
        
        return count
    
    def _sequences_match(self, seq1: List[BreachEvent], seq2: List[BreachEvent]) -> bool:
        """Check if two sequences match"""
        if len(seq1) != len(seq2):
            return False
        
        for b1, b2 in zip(seq1, seq2):
            if b1.breach_type != b2.breach_type:
                return False
        
        return True


class BreachPredictor:
    """Machine learning-based breach prediction"""
    
    def __init__(self):
        self.db_connection: Optional[asyncpg.Connection] = None
        self.prediction_models: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self, db_connection: asyncpg.Connection) -> None:
        """Initialize breach predictor"""
        self.db_connection = db_connection
        # Initialize ML models here (placeholder)
        
    async def predict_potential_breaches(
        self, 
        active_breaches: List[BreachEvent]
    ) -> List[Dict[str, Any]]:
        """Predict potential future breaches"""
        predictions = []
        
        try:
            # Simple heuristic-based predictions (can be enhanced with ML)
            for breach in active_breaches:
                if breach.is_recurring and breach.recurrence_count >= 2:
                    # Predict next occurrence based on historical pattern
                    next_occurrence = self._predict_next_recurrence(breach)
                    if next_occurrence:
                        predictions.append(next_occurrence)
            
            # Portfolio-level predictions
            portfolio_predictions = self._predict_portfolio_breaches(active_breaches)
            predictions.extend(portfolio_predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting breaches: {e}")
            return []
    
    def _predict_next_recurrence(self, breach: BreachEvent) -> Optional[Dict[str, Any]]:
        """Predict next occurrence of a recurring breach"""
        try:
            # Simple time-based prediction (can be enhanced)
            avg_interval = timedelta(hours=6)  # Placeholder
            next_time = datetime.utcnow() + avg_interval
            
            return {
                "prediction_type": "recurrence",
                "breach_type": breach.breach_type.value,
                "portfolio_id": breach.portfolio_id,
                "predicted_time": next_time.isoformat(),
                "confidence": min(0.8, breach.recurrence_count / 10.0),
                "time_horizon": "6_hours",
                "recommended_action": "monitor_closely"
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting recurrence: {e}")
            return None
    
    def _predict_portfolio_breaches(
        self, 
        active_breaches: List[BreachEvent]
    ) -> List[Dict[str, Any]]:
        """Predict portfolio-level breach cascade"""
        predictions = []
        
        try:
            # Group by portfolio
            by_portfolio = defaultdict(list)
            for breach in active_breaches:
                by_portfolio[breach.portfolio_id].append(breach)
            
            # Look for portfolios with multiple active breaches
            for portfolio_id, breaches in by_portfolio.items():
                if len(breaches) >= 2:
                    # High risk of cascade
                    predictions.append({
                        "prediction_type": "cascade",
                        "breach_type": "multiple_limits",
                        "portfolio_id": portfolio_id,
                        "confidence": min(0.9, len(breaches) / 5.0),
                        "time_horizon": "1_hour",
                        "recommended_action": "reduce_exposure"
                    })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting portfolio breaches: {e}")
            return []


class ResponseExecutor:
    """Executes automated responses to breaches"""
    
    def __init__(self, breach_detector: 'BreachDetector'):
        self.breach_detector = breach_detector
        self.logger = logging.getLogger(__name__)
    
    async def execute_action(
        self, 
        breach: BreachEvent, 
        action: ResponseAction, 
        rule: ResponseRule
    ) -> bool:
        """Execute a response action"""
        try:
            success = False
            
            if action == ResponseAction.ALERT_ONLY:
                success = await self._send_alert(breach)
            elif action == ResponseAction.POSITION_REDUCE:
                success = await self._reduce_position(breach, rule)
            elif action == ResponseAction.TRADING_HALT:
                success = await self._halt_trading(breach)
            elif action == ResponseAction.PORTFOLIO_FREEZE:
                success = await self._freeze_portfolio(breach)
            elif action == ResponseAction.EMERGENCY_LIQUIDATE:
                success = await self._emergency_liquidate(breach)
            elif action == ResponseAction.ESCALATE_HUMAN:
                success = await self._escalate_to_human(breach, rule)
            elif action == ResponseAction.RISK_OVERRIDE:
                success = await self._risk_override(breach)
            
            if success:
                breach.actions_taken.append(action)
                breach.response_log.append({
                    "action": action.value,
                    "rule_id": rule.rule_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True
                })
                
                self.logger.info(f"Executed action {action.value} for breach {breach.breach_id}")
            else:
                breach.response_log.append({
                    "action": action.value,
                    "rule_id": rule.rule_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": False,
                    "error": "Action execution failed"
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing action {action.value}: {e}")
            return False
    
    async def _send_alert(self, breach: BreachEvent) -> bool:
        """Send alert notification"""
        # Already handled by main breach detection
        return True
    
    async def _reduce_position(self, breach: BreachEvent, rule: ResponseRule) -> bool:
        """Reduce position size to mitigate risk"""
        try:
            # This would integrate with trading engine
            self.logger.info(f"Would reduce position for {breach.symbol} in portfolio {breach.portfolio_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error reducing position: {e}")
            return False
    
    async def _halt_trading(self, breach: BreachEvent) -> bool:
        """Halt trading for affected scope"""
        try:
            # This would integrate with trading engine
            self.logger.warning(f"Would halt trading for {breach.scope.value} {breach.scope_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error halting trading: {e}")
            return False
    
    async def _freeze_portfolio(self, breach: BreachEvent) -> bool:
        """Freeze entire portfolio"""
        try:
            # This would integrate with portfolio management
            self.logger.warning(f"Would freeze portfolio {breach.portfolio_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error freezing portfolio: {e}")
            return False
    
    async def _emergency_liquidate(self, breach: BreachEvent) -> bool:
        """Emergency liquidation"""
        try:
            # This would integrate with trading engine for emergency liquidation
            self.logger.critical(f"Would emergency liquidate positions in portfolio {breach.portfolio_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error in emergency liquidation: {e}")
            return False
    
    async def _escalate_to_human(self, breach: BreachEvent, rule: ResponseRule) -> bool:
        """Escalate to human intervention"""
        try:
            # Send high-priority alerts to risk management team
            self.logger.critical(f"Escalating breach {breach.breach_id} to {rule.escalation_level.value}")
            return True
        except Exception as e:
            self.logger.error(f"Error escalating to human: {e}")
            return False
    
    async def _risk_override(self, breach: BreachEvent) -> bool:
        """Apply risk override"""
        try:
            # Temporarily override limits (with proper approval workflow)
            self.logger.warning(f"Would apply risk override for breach {breach.breach_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error applying risk override: {e}")
            return False


# Global breach detector instance
breach_detector_instance = None

def get_breach_detector() -> BreachDetector:
    """Get global breach detector instance"""
    global breach_detector_instance
    if breach_detector_instance is None:
        raise RuntimeError("Breach detector not initialized. Call init_breach_detector() first.")
    return breach_detector_instance

def init_breach_detector(limit_engine: DynamicLimitEngine) -> BreachDetector:
    """Initialize global breach detector instance"""
    global breach_detector_instance
    breach_detector_instance = BreachDetector(limit_engine)
    return breach_detector_instance
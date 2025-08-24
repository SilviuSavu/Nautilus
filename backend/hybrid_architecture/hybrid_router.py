"""
Intelligent Hybrid Routing System for Nautilus Trading Platform
Combines direct engine access with enhanced gateway routing based on 
operation type, engine health, performance requirements, and load balancing.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from contextlib import asynccontextmanager

from .circuit_breaker import circuit_breaker_registry, CircuitBreakerOpenException
from .health_monitor import health_monitor, EngineStatus
from .enhanced_gateway import enhanced_gateway, RequestPriority, CacheStrategy

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy options"""
    DIRECT_ONLY = "direct_only"           # Direct engine access only
    GATEWAY_ONLY = "gateway_only"         # Enhanced gateway only
    HYBRID_PERFORMANCE = "hybrid_performance"  # Performance-based routing
    HYBRID_INTELLIGENT = "hybrid_intelligent"  # AI-driven routing decisions
    FAILOVER = "failover"                 # Direct with gateway failover


class OperationCategory(Enum):
    """Operation categories for routing decisions"""
    CRITICAL_TRADING = "critical_trading"      # <50ms requirement
    REAL_TIME_RISK = "real_time_risk"          # <100ms requirement
    ANALYTICS_STREAMING = "analytics_streaming" # <200ms requirement
    ML_INFERENCE = "ml_inference"              # <300ms requirement
    BACKGROUND_PROCESSING = "background_processing"  # <2000ms requirement
    USER_INTERFACE = "user_interface"          # <500ms requirement


@dataclass
class RoutingRule:
    """Routing rule configuration"""
    category: OperationCategory
    engine: str
    endpoint_pattern: str
    strategy: RoutingStrategy
    max_latency_ms: int
    priority: RequestPriority
    cache_strategy: CacheStrategy
    enable_fallback: bool = True
    load_balance: bool = False
    weight: float = 1.0


@dataclass
class RoutingDecision:
    """Routing decision result"""
    strategy: RoutingStrategy
    target_engine: str
    use_direct_access: bool
    use_gateway: bool
    enable_fallback: bool
    expected_latency_ms: int
    confidence: float
    reasoning: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class PerformanceProfile:
    """Engine performance profile"""
    engine: str
    direct_avg_latency_ms: float = 0
    direct_p95_latency_ms: float = 0
    direct_success_rate: float = 100
    gateway_avg_latency_ms: float = 0
    gateway_p95_latency_ms: float = 0
    gateway_success_rate: float = 100
    current_load: float = 0
    health_status: EngineStatus = EngineStatus.UNKNOWN
    last_updated: float = 0
    
    @property
    def direct_score(self) -> float:
        """Calculate direct access performance score"""
        latency_score = max(0, 100 - self.direct_avg_latency_ms / 10)
        success_score = self.direct_success_rate
        load_score = max(0, 100 - self.current_load)
        health_score = 100 if self.health_status == EngineStatus.HEALTHY else 50 if self.health_status == EngineStatus.DEGRADED else 0
        
        return (latency_score * 0.3 + success_score * 0.3 + load_score * 0.2 + health_score * 0.2)
    
    @property
    def gateway_score(self) -> float:
        """Calculate gateway performance score"""
        latency_score = max(0, 100 - self.gateway_avg_latency_ms / 10)
        success_score = self.gateway_success_rate
        # Gateway handles load balancing internally
        
        return (latency_score * 0.5 + success_score * 0.5)


class LoadBalancer:
    """
    Intelligent load balancer with multiple algorithms.
    Supports round-robin, least-connections, and performance-based routing.
    """
    
    def __init__(self):
        self.engine_loads: Dict[str, float] = {}
        self.engine_connections: Dict[str, int] = {}
        self.round_robin_counters: Dict[str, int] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def update_engine_load(self, engine: str, cpu_percent: float, memory_percent: float):
        """Update engine resource utilization"""
        # Composite load score
        load_score = (cpu_percent * 0.7 + memory_percent * 0.3)
        self.engine_loads[engine] = load_score
    
    def update_connection_count(self, engine: str, active_connections: int):
        """Update active connection count"""
        self.engine_connections[engine] = active_connections
    
    def record_performance(self, engine: str, response_time_ms: float):
        """Record performance metric"""
        if engine not in self.performance_history:
            self.performance_history[engine] = []
        
        self.performance_history[engine].append(response_time_ms)
        
        # Keep only last 100 samples
        if len(self.performance_history[engine]) > 100:
            self.performance_history[engine] = self.performance_history[engine][-100:]
    
    def select_engine_round_robin(self, engines: List[str]) -> str:
        """Select engine using round-robin algorithm"""
        if not engines:
            raise ValueError("No engines provided")
        
        engines_key = "|".join(sorted(engines))
        
        if engines_key not in self.round_robin_counters:
            self.round_robin_counters[engines_key] = 0
        
        index = self.round_robin_counters[engines_key] % len(engines)
        self.round_robin_counters[engines_key] += 1
        
        return engines[index]
    
    def select_engine_least_connections(self, engines: List[str]) -> str:
        """Select engine with least active connections"""
        if not engines:
            raise ValueError("No engines provided")
        
        return min(engines, key=lambda e: self.engine_connections.get(e, 0))
    
    def select_engine_performance_based(self, engines: List[str]) -> str:
        """Select engine based on performance metrics"""
        if not engines:
            raise ValueError("No engines provided")
        
        engine_scores = {}
        
        for engine in engines:
            # Calculate composite score
            load_score = 100 - self.engine_loads.get(engine, 50)
            connection_score = max(0, 100 - self.engine_connections.get(engine, 0) * 5)
            
            # Performance score based on average response time
            perf_history = self.performance_history.get(engine, [])
            if perf_history:
                avg_response_time = sum(perf_history) / len(perf_history)
                perf_score = max(0, 100 - avg_response_time / 10)
            else:
                perf_score = 50  # Default score for unknown performance
            
            engine_scores[engine] = (load_score * 0.4 + connection_score * 0.3 + perf_score * 0.3)
        
        # Select engine with highest score
        return max(engine_scores, key=engine_scores.get)


class RoutingMetrics:
    """Tracks routing decisions and performance metrics"""
    
    def __init__(self):
        self.decision_history: List[Dict[str, Any]] = []
        self.strategy_counts: Dict[RoutingStrategy, int] = {strategy: 0 for strategy in RoutingStrategy}
        self.latency_by_strategy: Dict[RoutingStrategy, List[float]] = {strategy: [] for strategy in RoutingStrategy}
        self.success_rates: Dict[RoutingStrategy, Dict[str, int]] = {strategy: {"success": 0, "total": 0} for strategy in RoutingStrategy}
        self.fallback_counts: Dict[str, int] = {}
        
    def record_decision(self, decision: RoutingDecision, actual_latency_ms: float, success: bool):
        """Record routing decision and outcome"""
        # Update strategy counts
        self.strategy_counts[decision.strategy] += 1
        
        # Record latency
        self.latency_by_strategy[decision.strategy].append(actual_latency_ms)
        if len(self.latency_by_strategy[decision.strategy]) > 1000:
            self.latency_by_strategy[decision.strategy] = self.latency_by_strategy[decision.strategy][-1000:]
        
        # Record success rates
        self.success_rates[decision.strategy]["total"] += 1
        if success:
            self.success_rates[decision.strategy]["success"] += 1
        
        # Store decision history
        self.decision_history.append({
            "timestamp": time.time(),
            "strategy": decision.strategy.value,
            "engine": decision.target_engine,
            "expected_latency_ms": decision.expected_latency_ms,
            "actual_latency_ms": actual_latency_ms,
            "success": success,
            "reasoning": decision.reasoning
        })
        
        # Keep only last 10000 decisions
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-10000:]
    
    def record_fallback(self, engine: str):
        """Record fallback usage"""
        self.fallback_counts[engine] = self.fallback_counts.get(engine, 0) + 1
    
    def get_strategy_performance(self, strategy: RoutingStrategy) -> Dict[str, Any]:
        """Get performance metrics for a routing strategy"""
        latencies = self.latency_by_strategy[strategy]
        success_data = self.success_rates[strategy]
        
        if not latencies:
            return {
                "total_requests": 0,
                "success_rate": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0
            }
        
        sorted_latencies = sorted(latencies)
        
        return {
            "total_requests": len(latencies),
            "success_rate": (success_data["success"] / success_data["total"]) * 100 if success_data["total"] > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 0 else 0,
            "p99_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 0 else 0
        }


class IntelligentHybridRouter:
    """
    Intelligent hybrid routing system that makes optimal routing decisions
    based on operation requirements, engine health, performance metrics, 
    and current system load.
    """
    
    def __init__(self):
        self.routing_rules = self._initialize_routing_rules()
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.load_balancer = LoadBalancer()
        self.metrics = RoutingMetrics()
        self.enabled = True
        self.default_strategy = RoutingStrategy.HYBRID_INTELLIGENT
        
        # Initialize performance profiles for all engines
        self._initialize_performance_profiles()
    
    def _initialize_routing_rules(self) -> List[RoutingRule]:
        """Initialize routing rules for different operation categories"""
        return [
            # Critical trading operations - direct access preferred
            RoutingRule(
                category=OperationCategory.CRITICAL_TRADING,
                engine="strategy",
                endpoint_pattern="/execute*",
                strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                max_latency_ms=50,
                priority=RequestPriority.CRITICAL,
                cache_strategy=CacheStrategy.NONE,
                enable_fallback=True,
                load_balance=True
            ),
            
            # Real-time risk calculations
            RoutingRule(
                category=OperationCategory.REAL_TIME_RISK,
                engine="risk",
                endpoint_pattern="/calculate*",
                strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                max_latency_ms=100,
                priority=RequestPriority.HIGH,
                cache_strategy=CacheStrategy.SHORT,
                enable_fallback=True,
                load_balance=True
            ),
            
            # Analytics streaming
            RoutingRule(
                category=OperationCategory.ANALYTICS_STREAMING,
                engine="analytics",
                endpoint_pattern="/real-time*",
                strategy=RoutingStrategy.HYBRID_INTELLIGENT,
                max_latency_ms=200,
                priority=RequestPriority.NORMAL,
                cache_strategy=CacheStrategy.MEDIUM,
                enable_fallback=True,
                load_balance=False
            ),
            
            # ML inference
            RoutingRule(
                category=OperationCategory.ML_INFERENCE,
                engine="ml",
                endpoint_pattern="/predict*",
                strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                max_latency_ms=300,
                priority=RequestPriority.HIGH,
                cache_strategy=CacheStrategy.SHORT,
                enable_fallback=True,
                load_balance=False
            ),
            
            # Background processing - gateway preferred for efficiency
            RoutingRule(
                category=OperationCategory.BACKGROUND_PROCESSING,
                engine="factor|features|marketdata|portfolio",
                endpoint_pattern="*",
                strategy=RoutingStrategy.GATEWAY_ONLY,
                max_latency_ms=2000,
                priority=RequestPriority.NORMAL,
                cache_strategy=CacheStrategy.LONG,
                enable_fallback=False,
                load_balance=True
            ),
            
            # User interface operations
            RoutingRule(
                category=OperationCategory.USER_INTERFACE,
                engine="websocket|portfolio",
                endpoint_pattern="*",
                strategy=RoutingStrategy.HYBRID_INTELLIGENT,
                max_latency_ms=500,
                priority=RequestPriority.NORMAL,
                cache_strategy=CacheStrategy.MEDIUM,
                enable_fallback=True,
                load_balance=True
            )
        ]
    
    def _initialize_performance_profiles(self):
        """Initialize performance profiles for all engines"""
        engines = ["strategy", "risk", "analytics", "ml", "factor", 
                  "features", "websocket", "marketdata", "portfolio"]
        
        for engine in engines:
            self.performance_profiles[engine] = PerformanceProfile(engine=engine)
    
    async def update_performance_profiles(self):
        """Update performance profiles from health monitor and gateway metrics"""
        try:
            # Get engine health from health monitor
            engine_health = await health_monitor.get_all_engine_health()
            
            for engine_name, health_metrics in engine_health.items():
                profile = self.performance_profiles.get(engine_name)
                if not profile:
                    continue
                
                profile.health_status = health_metrics.status
                profile.current_load = (health_metrics.cpu_percent + health_metrics.memory_percent) / 2
                profile.last_updated = time.time()
                
                # Update load balancer
                self.load_balancer.update_engine_load(
                    engine_name, 
                    health_metrics.cpu_percent, 
                    health_metrics.memory_percent
                )
            
            # Get gateway performance metrics
            gateway_status = await enhanced_gateway.get_gateway_status()
            
            # Update performance profiles with gateway metrics
            for engine_name in self.performance_profiles.keys():
                # This would be expanded with actual gateway metrics
                # For now, using placeholder values
                pass
            
        except Exception as e:
            logger.error(f"Error updating performance profiles: {e}")
    
    def _match_routing_rule(self, engine: str, endpoint: str) -> Optional[RoutingRule]:
        """Match request to routing rule"""
        for rule in self.routing_rules:
            # Check engine match (supports regex-like patterns)
            if "|" in rule.engine:
                engines = rule.engine.split("|")
                if engine not in engines:
                    continue
            elif rule.engine != engine and rule.engine != "*":
                continue
            
            # Check endpoint pattern match
            if rule.endpoint_pattern == "*":
                return rule
            elif rule.endpoint_pattern.endswith("*"):
                prefix = rule.endpoint_pattern[:-1]
                if endpoint.startswith(prefix):
                    return rule
            elif rule.endpoint_pattern.startswith("*"):
                suffix = rule.endpoint_pattern[1:]
                if endpoint.endswith(suffix):
                    return rule
            elif rule.endpoint_pattern == endpoint:
                return rule
        
        return None
    
    async def make_routing_decision(
        self,
        engine: str,
        endpoint: str,
        method: str = "GET",
        priority: Optional[RequestPriority] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Make intelligent routing decision based on multiple factors
        
        Args:
            engine: Target engine name
            endpoint: API endpoint
            method: HTTP method
            priority: Request priority override
            context: Additional context for decision making
            
        Returns:
            RoutingDecision with optimal routing strategy
        """
        
        if not self.enabled:
            return self._fallback_decision(engine, "Hybrid routing disabled")
        
        # Update performance profiles
        await self.update_performance_profiles()
        
        # Find matching routing rule
        rule = self._match_routing_rule(engine, endpoint)
        if not rule:
            return self._fallback_decision(engine, "No matching routing rule")
        
        # Get engine performance profile
        profile = self.performance_profiles.get(engine)
        if not profile:
            return self._fallback_decision(engine, "No performance profile available")
        
        # Override priority if specified
        effective_priority = priority or rule.priority
        
        # Make decision based on routing strategy
        if rule.strategy == RoutingStrategy.DIRECT_ONLY:
            return self._decide_direct_only(rule, profile)
        elif rule.strategy == RoutingStrategy.GATEWAY_ONLY:
            return self._decide_gateway_only(rule, profile)
        elif rule.strategy == RoutingStrategy.HYBRID_PERFORMANCE:
            return await self._decide_hybrid_performance(rule, profile, effective_priority)
        elif rule.strategy == RoutingStrategy.HYBRID_INTELLIGENT:
            return await self._decide_hybrid_intelligent(rule, profile, effective_priority, context)
        elif rule.strategy == RoutingStrategy.FAILOVER:
            return self._decide_failover(rule, profile)
        else:
            return self._fallback_decision(engine, "Unknown routing strategy")
    
    def _decide_direct_only(self, rule: RoutingRule, profile: PerformanceProfile) -> RoutingDecision:
        """Direct access only routing decision"""
        return RoutingDecision(
            strategy=RoutingStrategy.DIRECT_ONLY,
            target_engine=profile.engine,
            use_direct_access=True,
            use_gateway=False,
            enable_fallback=rule.enable_fallback,
            expected_latency_ms=int(profile.direct_avg_latency_ms) or 50,
            confidence=0.9 if profile.health_status == EngineStatus.HEALTHY else 0.3,
            reasoning="Direct access only per routing rule"
        )
    
    def _decide_gateway_only(self, rule: RoutingRule, profile: PerformanceProfile) -> RoutingDecision:
        """Gateway only routing decision"""
        return RoutingDecision(
            strategy=RoutingStrategy.GATEWAY_ONLY,
            target_engine=profile.engine,
            use_direct_access=False,
            use_gateway=True,
            enable_fallback=False,
            expected_latency_ms=int(profile.gateway_avg_latency_ms) or 100,
            confidence=0.8,
            reasoning="Enhanced gateway routing for efficiency and caching"
        )
    
    async def _decide_hybrid_performance(
        self, 
        rule: RoutingRule, 
        profile: PerformanceProfile,
        priority: RequestPriority
    ) -> RoutingDecision:
        """Performance-based hybrid routing decision"""
        
        # Critical operations require healthy engines for direct access
        if priority == RequestPriority.CRITICAL:
            if profile.health_status == EngineStatus.HEALTHY and profile.direct_score > 80:
                return RoutingDecision(
                    strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                    target_engine=profile.engine,
                    use_direct_access=True,
                    use_gateway=False,
                    enable_fallback=True,
                    expected_latency_ms=int(profile.direct_avg_latency_ms) or 30,
                    confidence=0.95,
                    reasoning="Direct access for critical operation with healthy engine"
                )
        
        # Performance comparison
        direct_score = profile.direct_score
        gateway_score = profile.gateway_score
        
        # Account for current load
        if profile.current_load > 80:
            gateway_score += 20  # Gateway handles load balancing better
        
        # Make decision based on scores
        if direct_score > gateway_score + 10:  # 10 point threshold for switching
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                target_engine=profile.engine,
                use_direct_access=True,
                use_gateway=False,
                enable_fallback=rule.enable_fallback,
                expected_latency_ms=int(profile.direct_avg_latency_ms) or 50,
                confidence=min(0.95, direct_score / 100),
                reasoning=f"Direct access chosen (score: {direct_score:.1f} vs {gateway_score:.1f})"
            )
        else:
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID_PERFORMANCE,
                target_engine=profile.engine,
                use_direct_access=False,
                use_gateway=True,
                enable_fallback=rule.enable_fallback,
                expected_latency_ms=int(profile.gateway_avg_latency_ms) or 100,
                confidence=min(0.9, gateway_score / 100),
                reasoning=f"Gateway chosen (score: {gateway_score:.1f} vs {direct_score:.1f})"
            )
    
    async def _decide_hybrid_intelligent(
        self,
        rule: RoutingRule,
        profile: PerformanceProfile,
        priority: RequestPriority,
        context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """AI-driven intelligent routing decision"""
        
        # Collect decision factors
        factors = {
            "health_status": profile.health_status.value,
            "current_load": profile.current_load,
            "direct_score": profile.direct_score,
            "gateway_score": profile.gateway_score,
            "priority": priority.value,
            "max_latency_ms": rule.max_latency_ms,
            "time_of_day": time.localtime().tm_hour,
            "context": context or {}
        }
        
        # Simple AI-like decision logic (can be replaced with ML model)
        decision_score = self._calculate_intelligent_score(factors)
        
        if decision_score > 0.7:
            # High confidence in direct access
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID_INTELLIGENT,
                target_engine=profile.engine,
                use_direct_access=True,
                use_gateway=False,
                enable_fallback=True,
                expected_latency_ms=int(profile.direct_avg_latency_ms) or 50,
                confidence=decision_score,
                reasoning=f"Intelligent routing: direct access (confidence: {decision_score:.2f})"
            )
        elif decision_score < 0.3:
            # High confidence in gateway
            return RoutingDecision(
                strategy=RoutingStrategy.HYBRID_INTELLIGENT,
                target_engine=profile.engine,
                use_direct_access=False,
                use_gateway=True,
                enable_fallback=rule.enable_fallback,
                expected_latency_ms=int(profile.gateway_avg_latency_ms) or 100,
                confidence=1.0 - decision_score,
                reasoning=f"Intelligent routing: gateway (confidence: {1.0-decision_score:.2f})"
            )
        else:
            # Medium confidence - use performance-based fallback
            return await self._decide_hybrid_performance(rule, profile, priority)
    
    def _calculate_intelligent_score(self, factors: Dict[str, Any]) -> float:
        """
        Calculate intelligent routing score (0-1 scale)
        Higher score = prefer direct access
        Lower score = prefer gateway
        """
        score = 0.5  # Start neutral
        
        # Health status factor
        if factors["health_status"] == "healthy":
            score += 0.3
        elif factors["health_status"] == "degraded":
            score += 0.1
        else:
            score -= 0.4
        
        # Load factor
        if factors["current_load"] < 50:
            score += 0.2
        elif factors["current_load"] > 80:
            score -= 0.2
        
        # Performance factor
        direct_vs_gateway = factors["direct_score"] - factors["gateway_score"]
        score += direct_vs_gateway / 500  # Normalize to roughly -0.2 to 0.2
        
        # Priority factor
        if factors["priority"] == "critical":
            score += 0.2
        elif factors["priority"] == "high":
            score += 0.1
        
        # Time-based factor (example: prefer direct access during trading hours)
        hour = factors["time_of_day"]
        if 9 <= hour <= 16:  # Trading hours
            score += 0.1
        
        # Constrain to 0-1 range
        return max(0.0, min(1.0, score))
    
    def _decide_failover(self, rule: RoutingRule, profile: PerformanceProfile) -> RoutingDecision:
        """Failover routing decision"""
        return RoutingDecision(
            strategy=RoutingStrategy.FAILOVER,
            target_engine=profile.engine,
            use_direct_access=True,
            use_gateway=False,
            enable_fallback=True,
            expected_latency_ms=int(profile.direct_avg_latency_ms) or 50,
            confidence=0.7,
            reasoning="Failover strategy: try direct access first, fallback to gateway"
        )
    
    def _fallback_decision(self, engine: str, reason: str) -> RoutingDecision:
        """Fallback routing decision when intelligent routing fails"""
        return RoutingDecision(
            strategy=RoutingStrategy.GATEWAY_ONLY,
            target_engine=engine,
            use_direct_access=False,
            use_gateway=True,
            enable_fallback=False,
            expected_latency_ms=200,
            confidence=0.5,
            reasoning=f"Fallback to gateway: {reason}"
        )
    
    def record_routing_outcome(
        self,
        decision: RoutingDecision,
        actual_latency_ms: float,
        success: bool
    ):
        """Record the outcome of a routing decision for learning"""
        self.metrics.record_decision(decision, actual_latency_ms, success)
        
        # Update load balancer performance tracking
        self.load_balancer.record_performance(decision.target_engine, actual_latency_ms)
        
        # Log interesting outcomes
        if not success:
            logger.warning(f"Routing failure: {decision.target_engine} - {decision.reasoning}")
        elif actual_latency_ms > decision.expected_latency_ms * 2:
            logger.warning(f"High latency: {actual_latency_ms}ms vs expected {decision.expected_latency_ms}ms")
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics"""
        strategy_metrics = {}
        for strategy in RoutingStrategy:
            strategy_metrics[strategy.value] = self.metrics.get_strategy_performance(strategy)
        
        return {
            "enabled": self.enabled,
            "total_decisions": sum(self.metrics.strategy_counts.values()),
            "strategy_distribution": {s.value: count for s, count in self.metrics.strategy_counts.items()},
            "strategy_performance": strategy_metrics,
            "fallback_counts": self.metrics.fallback_counts,
            "performance_profiles": {
                name: {
                    "health_status": profile.health_status.value,
                    "current_load": profile.current_load,
                    "direct_score": profile.direct_score,
                    "gateway_score": profile.gateway_score,
                    "last_updated": profile.last_updated
                }
                for name, profile in self.performance_profiles.items()
            }
        }
    
    def enable_routing(self):
        """Enable hybrid routing"""
        self.enabled = True
        logger.info("🔀 Hybrid routing enabled")
    
    def disable_routing(self):
        """Disable hybrid routing (fallback to gateway only)"""
        self.enabled = False
        logger.info("🔀 Hybrid routing disabled - using gateway fallback")


# Global hybrid router instance
hybrid_router = IntelligentHybridRouter()


# Convenience functions
async def route_request(
    engine: str,
    endpoint: str,
    method: str = "GET",
    priority: Optional[RequestPriority] = None,
    context: Optional[Dict[str, Any]] = None
) -> RoutingDecision:
    """Make routing decision for a request"""
    return await hybrid_router.make_routing_decision(engine, endpoint, method, priority, context)


def record_outcome(decision: RoutingDecision, latency_ms: float, success: bool):
    """Record routing outcome for learning"""
    hybrid_router.record_routing_outcome(decision, latency_ms, success)


async def get_routing_status() -> Dict[str, Any]:
    """Get comprehensive routing status"""
    return hybrid_router.get_routing_metrics()
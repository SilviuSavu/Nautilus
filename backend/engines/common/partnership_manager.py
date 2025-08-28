#!/usr/bin/env python3
"""
Partnership Manager
Manages engine relationships, tracks performance, and optimizes collaboration patterns.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict

from .engine_identity import EngineIdentity
from .engine_discovery import EngineDiscoveryProtocol
from .intelligent_router import MessageRouter, RouteMetrics


logger = logging.getLogger(__name__)


class PartnershipType(Enum):
    """Types of partnerships between engines"""
    PRIMARY = "primary"         # Critical dependency, high reliability required
    SECONDARY = "secondary"     # Important but not critical
    OPTIONAL = "optional"       # Nice to have, can function without
    COMPETITIVE = "competitive" # Multiple engines providing same capability
    COLLABORATIVE = "collaborative"  # Engines working together on tasks


class PartnershipStatus(Enum):
    """Status of partnerships"""
    PROPOSED = "proposed"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


@dataclass
class PartnershipAgreement:
    """Formal agreement between two engines"""
    partnership_id: str
    engine1_id: str
    engine2_id: str
    partnership_type: PartnershipType
    status: PartnershipStatus
    established_at: str
    
    # Performance expectations
    expected_latency_ms: float
    expected_throughput: int
    reliability_requirement: float  # 0.0-1.0
    
    # Communication preferences
    preferred_message_types: List[str]
    data_formats: List[str]
    communication_frequency: str  # "high", "medium", "low"
    
    # Relationship metadata
    relationship_strength: float = 0.5  # 0.0-1.0
    trust_score: float = 0.5           # 0.0-1.0
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    actual_latency_ms: float = 0.0
    actual_throughput: int = 0
    actual_reliability: float = 0.0
    message_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    last_interaction: Optional[str] = None
    last_evaluation: Optional[str] = None
    
    def __post_init__(self):
        if not self.partnership_id:
            self.partnership_id = str(uuid.uuid4())
    
    @property
    def success_rate(self) -> float:
        if self.message_count == 0:
            return 1.0
        return self.success_count / self.message_count
    
    @property
    def performance_score(self) -> float:
        """Overall performance score (0.0-1.0)"""
        latency_score = max(0, 1 - (self.actual_latency_ms / (self.expected_latency_ms * 2)))
        throughput_score = min(1, self.actual_throughput / max(1, self.expected_throughput))
        reliability_score = self.success_rate
        
        return (latency_score + throughput_score + reliability_score) / 3
    
    def update_performance(self, latency_ms: float, success: bool):
        """Update performance metrics"""
        alpha = 0.1  # Smoothing factor for exponential moving average
        
        # Update latency
        if self.actual_latency_ms == 0:
            self.actual_latency_ms = latency_ms
        else:
            self.actual_latency_ms = alpha * latency_ms + (1 - alpha) * self.actual_latency_ms
        
        # Update counters
        self.message_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update reliability
        self.actual_reliability = self.success_rate
        
        # Update relationship strength based on recent performance
        if success and latency_ms < self.expected_latency_ms * 1.5:
            self.relationship_strength = min(1.0, self.relationship_strength + 0.01)
        elif not success:
            self.relationship_strength = max(0.0, self.relationship_strength - 0.05)
        
        self.last_interaction = datetime.now().isoformat()
    
    def evaluate_health(self) -> Dict[str, Any]:
        """Evaluate partnership health"""
        health = "healthy"
        issues = []
        
        # Check latency
        if self.actual_latency_ms > self.expected_latency_ms * 2:
            health = "degraded"
            issues.append(f"High latency: {self.actual_latency_ms:.1f}ms > {self.expected_latency_ms * 2:.1f}ms")
        
        # Check reliability
        if self.actual_reliability < self.reliability_requirement:
            health = "degraded"
            issues.append(f"Low reliability: {self.actual_reliability:.2f} < {self.reliability_requirement:.2f}")
        
        # Check if partnership is stale
        if self.last_interaction:
            last_interaction = datetime.fromisoformat(self.last_interaction.replace('Z', '+00:00'))
            if datetime.now() - last_interaction > timedelta(hours=24):
                health = "stale"
                issues.append("No recent interaction")
        
        return {
            "health": health,
            "issues": issues,
            "performance_score": self.performance_score,
            "relationship_strength": self.relationship_strength
        }


@dataclass
class PartnershipRecommendation:
    """Recommendation for new or modified partnership"""
    target_engine_id: str
    recommended_type: PartnershipType
    confidence: float  # 0.0-1.0
    reasoning: List[str]
    expected_benefits: List[str]
    estimated_value: float  # Expected improvement score
    data_compatibility: float  # 0.0-1.0
    capability_complementarity: float  # 0.0-1.0


class PartnershipManager:
    """Manages all partnerships for an engine"""
    
    def __init__(
        self,
        engine_identity: EngineIdentity,
        discovery_protocol: EngineDiscoveryProtocol,
        message_router: MessageRouter
    ):
        self.identity = engine_identity
        self.discovery = discovery_protocol
        self.router = message_router
        
        # Partnership tracking
        self.partnerships: Dict[str, PartnershipAgreement] = {}
        self.partnership_proposals: Dict[str, Dict[str, Any]] = {}
        
        # Analytics
        self.collaboration_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.partnership_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._evaluation_interval = 300  # 5 minutes
        self._last_evaluation = datetime.now()
        
        # Recommendation system
        self._recommendation_cache: List[PartnershipRecommendation] = []
        self._last_recommendation_update = datetime.now()
    
    def create_partnership(
        self,
        partner_engine_id: str,
        partnership_type: PartnershipType,
        expected_latency_ms: float = 10.0,
        expected_throughput: int = 1000,
        reliability_requirement: float = 0.95
    ) -> PartnershipAgreement:
        """Create a new partnership agreement"""
        
        partnership = PartnershipAgreement(
            partnership_id=str(uuid.uuid4()),
            engine1_id=self.identity.engine_id,
            engine2_id=partner_engine_id,
            partnership_type=partnership_type,
            status=PartnershipStatus.ACTIVE,
            established_at=datetime.now().isoformat(),
            expected_latency_ms=expected_latency_ms,
            expected_throughput=expected_throughput,
            reliability_requirement=reliability_requirement,
            preferred_message_types=["GENERAL_COMMUNICATION"],
            data_formats=["json"],
            communication_frequency="medium"
        )
        
        self.partnerships[partner_engine_id] = partnership
        
        # Record in history
        self.partnership_history.append({
            "action": "created",
            "partner_id": partner_engine_id,
            "partnership_type": partnership_type.value,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Created {partnership_type.value} partnership with {partner_engine_id}")
        return partnership
    
    def update_partnership_performance(
        self,
        partner_engine_id: str,
        latency_ms: float,
        success: bool
    ):
        """Update partnership performance metrics"""
        if partner_engine_id in self.partnerships:
            partnership = self.partnerships[partner_engine_id]
            partnership.update_performance(latency_ms, success)
            
            # Record collaboration pattern
            self.collaboration_patterns[partner_engine_id].append({
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "success": success,
                "performance_score": partnership.performance_score
            })
            
            # Keep only recent history (last 1000 entries)
            if len(self.collaboration_patterns[partner_engine_id]) > 1000:
                self.collaboration_patterns[partner_engine_id] = \
                    self.collaboration_patterns[partner_engine_id][-1000:]
    
    def get_partnership(self, partner_engine_id: str) -> Optional[PartnershipAgreement]:
        """Get partnership with specific engine"""
        return self.partnerships.get(partner_engine_id)
    
    def get_partnerships_by_type(self, partnership_type: PartnershipType) -> List[PartnershipAgreement]:
        """Get all partnerships of specific type"""
        return [p for p in self.partnerships.values() if p.partnership_type == partnership_type]
    
    def get_active_partnerships(self) -> List[PartnershipAgreement]:
        """Get all active partnerships"""
        return [p for p in self.partnerships.values() if p.status == PartnershipStatus.ACTIVE]
    
    def evaluate_all_partnerships(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate health of all partnerships"""
        evaluations = {}
        
        for partner_id, partnership in self.partnerships.items():
            evaluation = partnership.evaluate_health()
            evaluations[partner_id] = evaluation
            
            # Update partnership status based on health
            if evaluation["health"] == "degraded":
                partnership.status = PartnershipStatus.DEGRADED
            elif evaluation["health"] == "stale":
                partnership.status = PartnershipStatus.SUSPENDED
            
            partnership.last_evaluation = datetime.now().isoformat()
        
        self._last_evaluation = datetime.now()
        return evaluations
    
    def find_partnership_opportunities(self) -> List[PartnershipRecommendation]:
        """Find new partnership opportunities"""
        if (datetime.now() - self._last_recommendation_update).seconds < 60:
            return self._recommendation_cache
        
        recommendations = []
        online_engines = self.discovery.get_online_engines()
        
        for engine_id, engine_data in online_engines.items():
            if engine_id == self.identity.engine_id or engine_id in self.partnerships:
                continue
            
            recommendation = self._analyze_partnership_potential(engine_id, engine_data)
            if recommendation and recommendation.confidence > 0.5:
                recommendations.append(recommendation)
        
        # Sort by estimated value
        recommendations.sort(key=lambda x: x.estimated_value, reverse=True)
        
        self._recommendation_cache = recommendations[:10]  # Top 10
        self._last_recommendation_update = datetime.now()
        
        return self._recommendation_cache
    
    def _analyze_partnership_potential(
        self,
        engine_id: str,
        engine_data: Dict[str, Any]
    ) -> Optional[PartnershipRecommendation]:
        """Analyze potential for partnership with specific engine"""
        
        # Check if engine is in preferred partners
        preferred_partners = self.identity.get_preferred_partners()
        is_preferred = engine_id in preferred_partners
        
        # Analyze data compatibility
        their_inputs = set(engine_data.get("input_formats", []))
        our_outputs = set([schema.format.value for schema in self.identity.capabilities.output_data_schemas])
        
        their_outputs = set(engine_data.get("output_formats", []))
        our_inputs = set([schema.format.value for schema in self.identity.capabilities.input_data_schemas])
        
        input_compatibility = len(their_inputs.intersection(our_outputs)) / max(len(our_outputs), 1)
        output_compatibility = len(their_outputs.intersection(our_inputs)) / max(len(our_inputs), 1)
        data_compatibility = (input_compatibility + output_compatibility) / 2
        
        # Analyze capability complementarity
        their_capabilities = set(engine_data.get("capabilities", []))
        our_capabilities = set([cap.value for cap in self.identity.capabilities.processing_capabilities])
        
        # Complementarity is better when capabilities don't overlap too much
        overlap = len(their_capabilities.intersection(our_capabilities))
        total_capabilities = len(their_capabilities.union(our_capabilities))
        capability_complementarity = 1 - (overlap / max(total_capabilities, 1))
        
        # Calculate confidence and value
        confidence = 0.0
        estimated_value = 0.0
        reasoning = []
        benefits = []
        
        if is_preferred:
            confidence += 0.4
            estimated_value += 0.3
            reasoning.append("Engine is in preferred partners list")
            benefits.append("Predefined compatibility and trust")
        
        if data_compatibility > 0.5:
            confidence += 0.3
            estimated_value += data_compatibility * 0.4
            reasoning.append(f"High data format compatibility ({data_compatibility:.2f})")
            benefits.append("Efficient data exchange")
        
        if capability_complementarity > 0.6:
            confidence += 0.2
            estimated_value += capability_complementarity * 0.2
            reasoning.append(f"Complementary capabilities ({capability_complementarity:.2f})")
            benefits.append("Mutual capability enhancement")
        
        # Check performance metrics
        performance = engine_data.get("performance", {})
        response_time = performance.get("response_time_ms", 1000)
        if response_time < 50:
            confidence += 0.1
            estimated_value += 0.1
            reasoning.append("Fast response time")
            benefits.append("Low latency communication")
        
        # Determine recommended partnership type
        recommended_type = PartnershipType.OPTIONAL
        if confidence > 0.8:
            recommended_type = PartnershipType.PRIMARY
        elif confidence > 0.6:
            recommended_type = PartnershipType.SECONDARY
        
        if confidence > 0.3:  # Only recommend if confidence is reasonable
            return PartnershipRecommendation(
                target_engine_id=engine_id,
                recommended_type=recommended_type,
                confidence=confidence,
                reasoning=reasoning,
                expected_benefits=benefits,
                estimated_value=estimated_value,
                data_compatibility=data_compatibility,
                capability_complementarity=capability_complementarity
            )
        
        return None
    
    def optimize_partnerships(self) -> Dict[str, Any]:
        """Optimize existing partnerships based on performance data"""
        optimizations = {
            "upgrades": [],
            "downgrades": [],
            "terminations": [],
            "modifications": []
        }
        
        for partner_id, partnership in self.partnerships.items():
            evaluation = partnership.evaluate_health()
            
            # Suggest upgrades for high-performing partnerships
            if (partnership.partnership_type == PartnershipType.SECONDARY and
                evaluation["performance_score"] > 0.9 and
                partnership.relationship_strength > 0.8):
                
                optimizations["upgrades"].append({
                    "partner_id": partner_id,
                    "current_type": partnership.partnership_type.value,
                    "suggested_type": PartnershipType.PRIMARY.value,
                    "reason": "Consistently high performance"
                })
            
            # Suggest downgrades for underperforming partnerships
            elif (partnership.partnership_type == PartnershipType.PRIMARY and
                  evaluation["performance_score"] < 0.6):
                
                optimizations["downgrades"].append({
                    "partner_id": partner_id,
                    "current_type": partnership.partnership_type.value,
                    "suggested_type": PartnershipType.SECONDARY.value,
                    "reason": "Declining performance"
                })
            
            # Suggest terminations for severely underperforming partnerships
            elif (evaluation["performance_score"] < 0.3 and
                  partnership.relationship_strength < 0.3):
                
                optimizations["terminations"].append({
                    "partner_id": partner_id,
                    "reason": "Consistently poor performance",
                    "performance_score": evaluation["performance_score"]
                })
        
        return optimizations
    
    def get_partnership_statistics(self) -> Dict[str, Any]:
        """Get comprehensive partnership statistics"""
        stats = {
            "total_partnerships": len(self.partnerships),
            "active_partnerships": len(self.get_active_partnerships()),
            "partnerships_by_type": {},
            "average_performance_score": 0.0,
            "top_performers": [],
            "collaboration_frequency": {},
            "recent_trends": {}
        }
        
        # Count by type
        for ptype in PartnershipType:
            stats["partnerships_by_type"][ptype.value] = len(self.get_partnerships_by_type(ptype))
        
        # Calculate average performance
        if self.partnerships:
            total_score = sum(p.performance_score for p in self.partnerships.values())
            stats["average_performance_score"] = total_score / len(self.partnerships)
        
        # Top performers
        sorted_partnerships = sorted(
            self.partnerships.items(),
            key=lambda x: x[1].performance_score,
            reverse=True
        )
        
        stats["top_performers"] = [
            {
                "partner_id": partner_id,
                "performance_score": partnership.performance_score,
                "relationship_strength": partnership.relationship_strength,
                "message_count": partnership.message_count
            }
            for partner_id, partnership in sorted_partnerships[:5]
        ]
        
        # Collaboration frequency
        for partner_id, partnership in self.partnerships.items():
            if partnership.last_interaction:
                last_interaction = datetime.fromisoformat(partnership.last_interaction.replace('Z', '+00:00'))
                hours_since = (datetime.now() - last_interaction).total_seconds() / 3600
                
                if hours_since < 1:
                    freq = "high"
                elif hours_since < 24:
                    freq = "medium"
                else:
                    freq = "low"
                
                stats["collaboration_frequency"][partner_id] = freq
        
        return stats
    
    async def auto_manage_partnerships(self):
        """Automatically manage partnerships based on performance"""
        # Evaluate all partnerships
        evaluations = self.evaluate_all_partnerships()
        
        # Get optimization suggestions
        optimizations = self.optimize_partnerships()
        
        # Apply automatic optimizations (for non-critical changes)
        for upgrade in optimizations["upgrades"]:
            partner_id = upgrade["partner_id"]
            if partner_id in self.partnerships:
                partnership = self.partnerships[partner_id]
                old_type = partnership.partnership_type
                partnership.partnership_type = PartnershipType.PRIMARY
                
                logger.info(f"Auto-upgraded partnership with {partner_id} from {old_type.value} to primary")
        
        # Log suggestions for manual review
        for downgrade in optimizations["downgrades"]:
            logger.warning(f"Partnership with {downgrade['partner_id']} may need downgrade: {downgrade['reason']}")
        
        for termination in optimizations["terminations"]:
            logger.error(f"Partnership with {termination['partner_id']} may need termination: {termination['reason']}")
        
        # Find new opportunities
        recommendations = self.find_partnership_opportunities()
        if recommendations:
            logger.info(f"Found {len(recommendations)} new partnership opportunities")
            for rec in recommendations[:3]:  # Log top 3
                logger.info(f"Recommendation: {rec.target_engine_id} ({rec.recommended_type.value}, confidence: {rec.confidence:.2f})")


if __name__ == "__main__":
    # Demo usage
    from .engine_identity import create_ml_engine_identity, create_risk_engine_identity
    
    async def demo():
        ml_engine = create_ml_engine_identity()
        discovery = EngineDiscoveryProtocol(ml_engine)
        router = MessageRouter(ml_engine, discovery)
        partnership_mgr = PartnershipManager(ml_engine, discovery, router)
        
        print("=== Partnership Manager Demo ===")
        
        # Create a partnership
        partnership = partnership_mgr.create_partnership(
            "RISK_ENGINE",
            PartnershipType.PRIMARY,
            expected_latency_ms=5.0,
            reliability_requirement=0.99
        )
        
        print(f"Created partnership: {partnership.partnership_id}")
        
        # Simulate some interactions
        for i in range(10):
            success = i < 8  # 80% success rate
            latency = 3.0 + (i % 3)  # Variable latency
            partnership_mgr.update_partnership_performance("RISK_ENGINE", latency, success)
        
        # Get statistics
        stats = partnership_mgr.get_partnership_statistics()
        print(f"Partnership statistics: {json.dumps(stats, indent=2)}")
        
        # Get partnership evaluation
        evaluations = partnership_mgr.evaluate_all_partnerships()
        print(f"Partnership evaluations: {json.dumps(evaluations, indent=2)}")
    
    asyncio.run(demo())
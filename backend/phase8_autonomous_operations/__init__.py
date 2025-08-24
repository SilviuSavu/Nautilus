"""
Nautilus Phase 8: Autonomous AI Operations & Self-Healing Systems

This module provides enterprise-grade autonomous operations with:
- 99.99% autonomous decision-making
- Predictive self-healing infrastructure
- Intelligent system optimization
- Autonomous incident response
- AI-driven resource allocation

All systems work together to provide minimal human intervention 
while maintaining maximum system reliability and performance.
"""

from .ai_operations.autonomous_ai_operations_center import AutonomousAIOperationsCenter
from .self_healing.predictive_self_healing_system import PredictiveSelfHealingSystem
from .optimization.intelligent_system_optimizer import IntelligentSystemOptimizer
from .incident_response.autonomous_incident_response import AutonomousIncidentResponseSystem
from .resource_allocation.adaptive_resource_allocator import AdaptiveResourceAllocator

__version__ = "1.0.0"
__author__ = "Nautilus Trading Platform"

__all__ = [
    "AutonomousAIOperationsCenter",
    "PredictiveSelfHealingSystem", 
    "IntelligentSystemOptimizer",
    "AutonomousIncidentResponseSystem",
    "AdaptiveResourceAllocator"
]
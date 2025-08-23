"""
Breach Detector Integration Example
Sprint 3: Comprehensive Risk Management

This example demonstrates how to integrate and use the breach detector
with the existing limit engine and risk monitoring infrastructure.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from .limit_engine import (
    DynamicLimitEngine, 
    LimitType, 
    LimitScope, 
    LimitStatus, 
    init_limit_engine
)
from .breach_detector import (
    BreachDetector,
    BreachEvent,
    BreachPattern,
    ResponseRule,
    ResponseAction,
    BreachSeverity,
    BreachCategory,
    EscalationLevel,
    init_breach_detector
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_breach_detection():
    """
    Complete demonstration of breach detection system
    
    Shows:
    1. Initialization and setup
    2. Creating limits and detecting breaches
    3. Automated response workflow
    4. Pattern detection
    5. Statistics and reporting
    """
    
    print("\n🚨 BREACH DETECTOR DEMONSTRATION - Sprint 3")
    print("=" * 60)
    
    # Initialize limit engine
    print("\n1. Initializing Limit Engine...")
    limit_engine = init_limit_engine()
    await limit_engine.initialize()
    
    # Initialize breach detector
    print("\n2. Initializing Breach Detector...")
    breach_detector = init_breach_detector(limit_engine)
    await breach_detector.initialize()
    
    # Setup breach event callbacks
    breach_detector.add_breach_callback(lambda breach: print(f"🔥 BREACH CALLBACK: {breach.breach_id} - {breach.severity.value}"))
    breach_detector.add_pattern_callback(lambda pattern: print(f"🔍 PATTERN CALLBACK: {pattern.pattern_id} - {pattern.pattern_type}"))
    
    try:
        # Create some risk limits
        await create_sample_limits(limit_engine)
        
        # Simulate breach scenarios
        await simulate_breach_scenarios(limit_engine, breach_detector)
        
        # Demonstrate pattern detection
        await demonstrate_pattern_detection(breach_detector)
        
        # Show breach management features
        await demonstrate_breach_management(breach_detector)
        
        # Display statistics
        await show_statistics(breach_detector)
        
        # Test response rules
        await demonstrate_response_rules(breach_detector)
        
    finally:
        # Cleanup
        print("\n9. Shutting down...")
        await breach_detector.shutdown()
        await limit_engine.shutdown()
        print("✅ Demonstration complete!")


async def create_sample_limits(limit_engine: DynamicLimitEngine):
    """Create sample risk limits for demonstration"""
    print("\n3. Creating Sample Risk Limits...")
    
    # Portfolio VaR limits
    var_limit_id = await limit_engine.create_limit(
        limit_type=LimitType.VAR_ABSOLUTE,
        scope=LimitScope.PORTFOLIO,
        scope_id="demo_portfolio_001",
        limit_value=Decimal("10000.00"),
        description="Daily VaR limit for demo portfolio",
        warning_threshold=0.8,
        breach_threshold=1.0
    )
    print(f"   ✓ Created VaR limit: {var_limit_id}")
    
    # Exposure limits
    exposure_limit_id = await limit_engine.create_limit(
        limit_type=LimitType.EXPOSURE_GROSS,
        scope=LimitScope.PORTFOLIO,
        scope_id="demo_portfolio_001",
        limit_value=Decimal("50000.00"),
        description="Gross exposure limit",
        warning_threshold=0.85,
        breach_threshold=1.0
    )
    print(f"   ✓ Created exposure limit: {exposure_limit_id}")
    
    # Concentration limits
    concentration_limit_id = await limit_engine.create_limit(
        limit_type=LimitType.CONCENTRATION,
        scope=LimitScope.SYMBOL,
        scope_id="AAPL",
        limit_value=Decimal("15.0"),  # 15% max concentration
        description="Single position concentration limit",
        warning_threshold=0.8,
        breach_threshold=1.0
    )
    print(f"   ✓ Created concentration limit: {concentration_limit_id}")
    
    # Drawdown limits
    drawdown_limit_id = await limit_engine.create_limit(
        limit_type=LimitType.DRAWDOWN_MAX,
        scope=LimitScope.PORTFOLIO,
        scope_id="demo_portfolio_001",
        limit_value=Decimal("5.0"),  # 5% max drawdown
        description="Maximum drawdown limit",
        warning_threshold=0.7,
        breach_threshold=1.0
    )
    print(f"   ✓ Created drawdown limit: {drawdown_limit_id}")


async def simulate_breach_scenarios(
    limit_engine: DynamicLimitEngine, 
    breach_detector: BreachDetector
):
    """Simulate various breach scenarios"""
    print("\n4. Simulating Breach Scenarios...")
    
    # Scenario 1: Minor VaR breach
    print("\n   📊 Scenario 1: Minor VaR Breach")
    var_checks = await simulate_limit_breach(
        limit_engine, 
        LimitType.VAR_ABSOLUTE, 
        current_value=Decimal("11000.00"),  # 110% of limit
        description="Market stress increases VaR"
    )
    if var_checks:
        breaches = await breach_detector.check_for_breaches(
            var_checks, 
            {"market_volatility": 0.35, "market_direction": "down"}
        )
        print(f"      → Detected {len(breaches)} breaches")
    
    await asyncio.sleep(1)  # Pause for demonstration
    
    # Scenario 2: Critical exposure breach
    print("\n   📊 Scenario 2: Critical Exposure Breach")
    exposure_checks = await simulate_limit_breach(
        limit_engine,
        LimitType.EXPOSURE_GROSS,
        current_value=Decimal("76000.00"),  # 152% of limit
        description="Large position accumulation"
    )
    if exposure_checks:
        breaches = await breach_detector.check_for_breaches(
            exposure_checks,
            {"trading_session": "us_regular", "liquidity": "stressed"}
        )
        print(f"      → Detected {len(breaches)} breaches")
    
    await asyncio.sleep(1)
    
    # Scenario 3: Emergency drawdown breach
    print("\n   📊 Scenario 3: Emergency Drawdown Breach")
    drawdown_checks = await simulate_limit_breach(
        limit_engine,
        LimitType.DRAWDOWN_MAX,
        current_value=Decimal("12.5"),  # 250% of limit - emergency level
        description="Severe market crash scenario"
    )
    if drawdown_checks:
        breaches = await breach_detector.check_for_breaches(
            drawdown_checks,
            {"market_volatility": 0.65, "market_direction": "crash"}
        )
        print(f"      → Detected {len(breaches)} breaches")


async def simulate_limit_breach(
    limit_engine: DynamicLimitEngine,
    limit_type: LimitType,
    current_value: Decimal,
    description: str
) -> List:
    """Helper to simulate a limit breach"""
    try:
        # Find a limit of the specified type
        target_limit = None
        for limit in limit_engine.limits.values():
            if limit.limit_type == limit_type:
                target_limit = limit
                break
        
        if not target_limit:
            print(f"      ⚠️ No {limit_type.value} limit found")
            return []
        
        # Create mock limit check showing breach
        from .limit_engine import LimitCheck
        
        utilization = float(current_value / target_limit.limit_value)
        
        if utilization >= target_limit.breach_threshold:
            status = LimitStatus.BREACHED
        elif utilization >= target_limit.warning_threshold:
            status = LimitStatus.WARNING
        else:
            status = LimitStatus.ACTIVE
        
        breach_amount = max(Decimal("0"), current_value - target_limit.limit_value)
        
        check = LimitCheck(
            limit_id=target_limit.limit_id,
            current_value=current_value,
            limit_value=target_limit.limit_value,
            utilization=utilization,
            status=status,
            breach_amount=breach_amount,
            timestamp=datetime.utcnow(),
            metadata={"description": description}
        )
        
        print(f"      💥 {description}: {current_value} vs limit {target_limit.limit_value} ({utilization:.1%})")
        
        return [check]
        
    except Exception as e:
        logger.error(f"Error simulating breach: {e}")
        return []


async def demonstrate_pattern_detection(breach_detector: BreachDetector):
    """Demonstrate pattern detection capabilities"""
    print("\n5. Demonstrating Pattern Detection...")
    
    # Simulate recurring breaches to trigger pattern detection
    print("\n   🔍 Simulating recurring breach pattern...")
    
    for i in range(4):  # Create 4 similar breaches
        # Create mock breach events
        from .breach_detector import BreachEvent, BreachSeverity, BreachCategory, BreachStatus
        from .limit_engine import LimitType, LimitScope, TimeWindow
        from decimal import Decimal
        import uuid
        
        breach = BreachEvent(
            breach_id=f"pattern_demo_{i}_{uuid.uuid4().hex[:8]}",
            limit_id="demo_var_limit",
            portfolio_id="demo_portfolio_001",
            strategy_id="momentum_strategy",
            symbol=None,
            breach_type=LimitType.VAR_ABSOLUTE,
            severity=BreachSeverity.MINOR if i < 2 else BreachSeverity.MAJOR,
            category=BreachCategory.LIMIT_VIOLATION,
            status=BreachStatus.DETECTED,
            current_value=Decimal("11000.00") + Decimal(str(i * 1000)),
            limit_value=Decimal("10000.00"),
            breach_amount=Decimal("1000.00") + Decimal(str(i * 1000)),
            breach_percentage=10.0 + i * 10,
            utilization_ratio=1.1 + i * 0.1,
            scope=LimitScope.PORTFOLIO,
            scope_id="demo_portfolio_001",
            time_window=TimeWindow.INTRADAY,
            market_conditions={"volatility": 0.3 + i * 0.05},
            actions_taken=[],
            escalation_level=EscalationLevel.TRADER,
            detected_at=datetime.utcnow() - timedelta(minutes=30 - i * 5)  # Space breaches 5 min apart
        )
        
        # Add to breach history to trigger pattern analysis
        breach_detector.breach_history.append(breach)
        breach_detector.active_breaches[breach.breach_id] = breach
        
        print(f"      + Added breach {i+1}: {breach.severity.value} at {breach.detected_at.strftime('%H:%M:%S')}")
        
        await asyncio.sleep(0.5)  # Small delay for realism
    
    # Trigger pattern analysis
    await breach_detector.pattern_detector.analyze_recent_breaches(list(breach_detector.breach_history))
    
    patterns = await breach_detector.get_breach_patterns()
    print(f"\n   📈 Detected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"      → Pattern {pattern.pattern_id}: {pattern.pattern_type} "
              f"(confidence: {pattern.confidence:.1%}, frequency: {pattern.frequency})")


async def demonstrate_breach_management(breach_detector: BreachDetector):
    """Demonstrate breach management features"""
    print("\n6. Demonstrating Breach Management...")
    
    active_breaches = await breach_detector.get_active_breaches()
    if not active_breaches:
        print("   📋 No active breaches to manage")
        return
    
    # Take first breach for demonstration
    breach = active_breaches[0]
    print(f"\n   🔧 Managing breach: {breach.breach_id}")
    
    # Acknowledge breach
    acknowledged = await breach_detector.acknowledge_breach(
        breach.breach_id, 
        "demo_risk_manager"
    )
    if acknowledged:
        print("      ✓ Breach acknowledged by risk manager")
    
    # Show updated status
    updated_breach = breach_detector.active_breaches.get(breach.breach_id)
    if updated_breach:
        print(f"      → Status: {updated_breach.status.value}")
        print(f"      → Acknowledged by: {updated_breach.acknowledged_by}")
        print(f"      → Response log entries: {len(updated_breach.response_log)}")
    
    # Resolve breach (simulated)
    await asyncio.sleep(1)  # Simulate time passing
    
    resolved = await breach_detector.resolve_breach(
        breach.breach_id,
        "demo_trader"
    )
    if resolved:
        print("      ✓ Breach resolved by trader")


async def show_statistics(breach_detector: BreachDetector):
    """Display breach detection statistics"""
    print("\n7. Breach Detection Statistics...")
    
    stats = breach_detector.get_breach_statistics()
    
    print(f"\n   📊 Overall Statistics:")
    print(f"      • Total breaches detected: {stats['total_breaches_detected']}")
    print(f"      • Active breaches: {stats['active_breaches']}")
    print(f"      • Breach patterns detected: {stats['breach_patterns_detected']}")
    print(f"      • Response rules active: {stats['response_rules_active']}")
    
    print(f"\n   📈 Breaches by Severity:")
    for severity, count in stats['breaches_by_severity'].items():
        print(f"      • {severity.value.capitalize()}: {count}")
    
    print(f"\n   📋 Most Common Breach Types:")
    for breach_type, count in stats['most_common_breach_types'][:3]:
        print(f"      • {breach_type.value}: {count}")
    
    # Show breach history
    history = await breach_detector.get_breach_history(hours_back=1)
    print(f"\n   🕐 Recent Breach History (last hour): {len(history)} breaches")
    for breach in history[:5]:  # Show first 5
        print(f"      • {breach.detected_at.strftime('%H:%M:%S')} - "
              f"{breach.breach_type.value} - {breach.severity.value}")


async def demonstrate_response_rules(breach_detector: BreachDetector):
    """Demonstrate response rule configuration and execution"""
    print("\n8. Demonstrating Response Rules...")
    
    # Create custom response rule
    custom_rule = ResponseRule(
        rule_id="demo_custom_rule",
        name="Demo Custom Response Rule",
        description="Custom rule for demonstration",
        severity_threshold=BreachSeverity.MAJOR,
        breach_types={LimitType.VAR_ABSOLUTE, LimitType.EXPOSURE_GROSS},
        categories={BreachCategory.LIMIT_VIOLATION},
        immediate_actions=[ResponseAction.ALERT_ONLY],
        escalation_delay_minutes=5,
        escalation_actions=[ResponseAction.POSITION_REDUCE, ResponseAction.ESCALATE_HUMAN],
        escalation_level=EscalationLevel.RISK_MANAGER
    )
    
    # Add rule
    added = await breach_detector.add_response_rule(custom_rule)
    if added:
        print("   ✓ Added custom response rule")
    
    # Show current rules
    print(f"\n   📋 Active Response Rules: {len(breach_detector.response_rules)}")
    for rule_id, rule in list(breach_detector.response_rules.items())[:3]:  # Show first 3
        print(f"      • {rule.name}: {rule.severity_threshold.value}+ → {len(rule.immediate_actions)} actions")
    
    # Show how rules would be applied
    active_breaches = await breach_detector.get_active_breaches()
    if active_breaches:
        breach = active_breaches[0]
        applicable_rules = breach_detector._find_applicable_rules(breach)
        print(f"\n   🎯 Applicable rules for breach {breach.breach_id}: {len(applicable_rules)}")
        for rule in applicable_rules:
            print(f"      • {rule.name}: {len(rule.immediate_actions)} immediate actions")


async def main():
    """Main demonstration function"""
    try:
        await demo_breach_detection()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
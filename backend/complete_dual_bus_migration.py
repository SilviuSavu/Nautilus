#!/usr/bin/env python3
"""
Complete Dual Bus Migration Plan
Migrate ALL engines to true dual bus architecture using Risk Engine as template
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class DualBusMigrationPlan:
    """Complete dual bus migration for all engines"""
    
    def __init__(self):
        self.engines_to_migrate = [
            # Currently single/enhanced messagebus - need full dual bus
            {"name": "Factor Engine", "port": 8300, "path": "engines/factor", "current": "enhanced_messagebus"},
            {"name": "Features Engine", "port": 8500, "path": "engines/features", "current": "enhanced_messagebus"}, 
            {"name": "WebSocket Engine", "port": 8600, "path": "engines/websocket", "current": "enhanced_messagebus"},
            {"name": "Portfolio Engine", "port": 8900, "path": "engines/portfolio", "current": "enhanced_messagebus"},
            {"name": "Collateral Engine", "port": 9000, "path": "engines/collateral", "current": "enhanced_messagebus"},
            
            # Pseudo dual bus - fix implementation
            {"name": "Analytics Engine", "port": 8100, "path": "engines/analytics", "current": "pseudo_dual_bus"},
            
            # No messagebus or basic - need full dual bus
            {"name": "Backtesting Engine", "port": 8110, "path": "engines/backtesting", "current": "basic_messagebus"},
            {"name": "MarketData Engine", "port": 8800, "path": "engines/marketdata", "current": "no_messagebus"},
            {"name": "VPIN Engine", "port": 10000, "path": "engines/vpin", "current": "basic_messagebus"},
            {"name": "Enhanced VPIN Engine", "port": 10001, "path": "engines/vpin", "current": "basic_messagebus"},
            
            # Not responding - need restart + dual bus
            {"name": "ML Engine", "port": 8400, "path": "engines/ml", "current": "not_responding"},
            {"name": "Strategy Engine", "port": 8700, "path": "engines/strategy", "current": "not_responding"},
        ]
        
        # Reference implementation - Risk Engine has working dual bus
        self.reference_engine = {
            "name": "Risk Engine", 
            "port": 8200, 
            "path": "engines/risk",
            "implementation": "dual_bus_risk_engine.py"
        }
    
    def analyze_current_state(self):
        """Analyze current dual bus implementation state"""
        print("🔍 DUAL BUS MIGRATION ANALYSIS")
        print("=" * 60)
        
        print("\n✅ REFERENCE IMPLEMENTATION (Working Dual Bus):")
        print(f"   {self.reference_engine['name']} - Port {self.reference_engine['port']}")
        print("   Architecture: dual_bus")  
        print("   MarketData Bus: 6380")
        print("   Engine Logic Bus: 6381")
        print("   Status: ✅ FULLY OPERATIONAL")
        
        print(f"\n❌ ENGINES NEEDING DUAL BUS MIGRATION ({len(self.engines_to_migrate)}):")
        for engine in self.engines_to_migrate:
            status_icon = "🔄" if engine["current"] != "not_responding" else "❌"
            print(f"   {status_icon} {engine['name']} - Port {engine['port']} - Current: {engine['current']}")
        
        print(f"\n📊 MIGRATION SCOPE:")
        print(f"   Engines needing migration: {len(self.engines_to_migrate)}")
        print(f"   Reference implementation: 1 (Risk Engine)")
        print(f"   Total engines after migration: {len(self.engines_to_migrate) + 1}")
        
        return self.engines_to_migrate
    
    def create_migration_strategy(self):
        """Create systematic migration strategy"""
        print("\n🚀 DUAL BUS MIGRATION STRATEGY")
        print("=" * 45)
        
        # Group engines by migration complexity
        groups = {
            "Phase 1 - Enhanced MessageBus (Easiest)": [],
            "Phase 2 - Pseudo Dual Bus Fix (Medium)": [],
            "Phase 3 - Basic/No MessageBus (Hard)": [],
            "Phase 4 - Not Responding (Hardest)": []
        }
        
        for engine in self.engines_to_migrate:
            if engine["current"] == "enhanced_messagebus":
                groups["Phase 1 - Enhanced MessageBus (Easiest)"].append(engine)
            elif engine["current"] == "pseudo_dual_bus":
                groups["Phase 2 - Pseudo Dual Bus Fix (Medium)"].append(engine)
            elif engine["current"] in ["basic_messagebus", "no_messagebus"]:
                groups["Phase 3 - Basic/No MessageBus (Hard)"].append(engine)
            elif engine["current"] == "not_responding":
                groups["Phase 4 - Not Responding (Hardest)"].append(engine)
        
        for phase, engines in groups.items():
            if engines:
                print(f"\n{phase}:")
                for engine in engines:
                    print(f"   • {engine['name']} ({engine['current']})")
        
        return groups
    
    def get_migration_template(self):
        """Get the dual bus implementation template from Risk Engine"""
        template = f"""
# Dual Bus Implementation Template (from Risk Engine)
# ================================================

from dual_messagebus_client import get_dual_bus_client, EngineType

class EngineDualBusIntegration:
    def __init__(self, engine_type: EngineType):
        self.engine_type = engine_type
        self.dual_client = None
        
    async def initialize_dual_bus(self):
        '''Initialize dual messagebus connection'''
        self.dual_client = await get_dual_bus_client(self.engine_type)
        
    async def publish_to_marketdata_bus(self, data):
        '''Publish to MarketData Bus (6380)'''
        await self.dual_client.publish_message(MessageType.MARKET_DATA, data)
        
    async def publish_to_engine_logic_bus(self, data):
        '''Publish to Engine Logic Bus (6381)'''  
        await self.dual_client.publish_message(MessageType.ENGINE_LOGIC, data)
        
    def get_health_status(self):
        '''Return dual bus health status'''
        return {{
            "architecture": "dual_bus",
            "marketdata_bus": "6380", 
            "engine_logic_bus": "6381",
            "dual_bus_connected": bool(self.dual_client)
        }}
        """
        
        return template
    
    def estimate_migration_effort(self):
        """Estimate migration effort and timeline"""
        print("\n⏱️  MIGRATION EFFORT ESTIMATION")
        print("=" * 35)
        
        effort_map = {
            "enhanced_messagebus": {"effort": "Low", "time": "30 min", "complexity": 2},
            "pseudo_dual_bus": {"effort": "Medium", "time": "45 min", "complexity": 3},
            "basic_messagebus": {"effort": "High", "time": "60 min", "complexity": 4},
            "no_messagebus": {"effort": "High", "time": "90 min", "complexity": 5},
            "not_responding": {"effort": "Very High", "time": "120 min", "complexity": 6}
        }
        
        total_time = 0
        total_complexity = 0
        
        for engine in self.engines_to_migrate:
            effort_info = effort_map[engine["current"]]
            print(f"   {engine['name']}: {effort_info['effort']} ({effort_info['time']})")
            total_time += int(effort_info['time'].split()[0])
            total_complexity += effort_info['complexity']
        
        print(f"\n📊 TOTAL ESTIMATES:")
        print(f"   Total time needed: ~{total_time} minutes ({total_time//60}h {total_time%60}m)")
        print(f"   Average complexity: {total_complexity/len(self.engines_to_migrate):.1f}/6")
        print(f"   Recommended approach: Phase-by-phase migration")
        
        return total_time, total_complexity

def main():
    """Main migration analysis"""
    migration_plan = DualBusMigrationPlan()
    
    # Analyze current state
    engines_to_migrate = migration_plan.analyze_current_state()
    
    # Create strategy
    migration_groups = migration_plan.create_migration_strategy()
    
    # Estimate effort
    total_time, complexity = migration_plan.estimate_migration_effort()
    
    # Show template
    print("\n📋 NEXT STEPS:")
    print("=" * 15)
    print("1. Start with Phase 1 engines (Enhanced MessageBus)")
    print("2. Use Risk Engine as implementation reference")
    print("3. Test each engine after migration")
    print("4. Validate dual bus connectivity")
    
    return migration_groups

if __name__ == "__main__":
    main()
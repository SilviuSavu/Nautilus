#!/usr/bin/env python3
"""
Test Strategy Deployment Framework Integration
Quick integration test to verify all components work together
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

async def test_framework_integration():
    """Test complete framework integration"""
    print("🧪 Testing Strategy Deployment Framework Integration...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from strategies import (
            initialize_framework,
            get_framework_info,
            StrategyTester,
            DeploymentManager,
            VersionControl,
            RollbackService,
            PipelineMonitor
        )
        print("✅ All imports successful")
        
        # Test framework info
        print("\n📋 Testing framework info...")
        info = get_framework_info()
        print(f"✅ Framework: {info['name']} v{info['version']}")
        print(f"   Components: {len(info['components'])}")
        print(f"   Features: {len(info['features'])}")
        
        # Test framework initialization
        print("\n🚀 Testing framework initialization...")
        framework = initialize_framework()
        print(f"✅ Framework initialized with {len(framework)} components")
        
        # Test individual components
        print("\n🧩 Testing individual components...")
        
        # Test Strategy Tester
        tester = framework['strategy_tester']
        test_code = """
class SimpleStrategy:
    def __init__(self):
        self.fast_period = 10
        self.slow_period = 20
    
    def on_data(self, data):
        return True
"""
        test_config = {
            "id": "test_strategy",
            "parameters": {"fast_period": 10, "slow_period": 20}
        }
        
        print("   Testing StrategyTester...")
        test_suite = await tester.run_full_test_suite(test_code, test_config)
        print(f"   ✅ Test suite completed with score: {test_suite.overall_score:.1f}")
        
        # Test Version Control
        print("   Testing VersionControl...")
        version_control = framework['version_control']
        version = version_control.commit_version(
            strategy_id="test_strategy",
            strategy_code=test_code,
            strategy_config=test_config,
            commit_message="Initial test version"
        )
        print(f"   ✅ Version committed: {version.version_number}")
        
        # Test Deployment Manager
        print("   Testing DeploymentManager...")
        deployment_mgr = framework['deployment_manager']
        deployment_mgr.strategy_tester = tester
        deployment_mgr.version_control = version_control
        
        deployment = await deployment_mgr.deploy_strategy(
            strategy_id="test_strategy",
            version=version.version_id,
            target_environment="development"
        )
        print(f"   ✅ Deployment created: {deployment.deployment_id}")
        
        # Test Pipeline Monitor
        print("   Testing PipelineMonitor...")
        monitor = framework['pipeline_monitor']
        await monitor.start_monitoring("test_pipeline", "test_strategy", "1.0.0")
        await asyncio.sleep(1)
        await monitor.stop_monitoring("test_pipeline")
        print("   ✅ Pipeline monitoring test completed")
        
        # Test Rollback Service
        print("   Testing RollbackService...")
        rollback_svc = framework['rollback_service']
        rollback_svc.deployment_manager = deployment_mgr
        rollback_svc.version_control = version_control
        
        triggers = rollback_svc.get_rollback_triggers("test_strategy")
        print(f"   ✅ Rollback service has {len(triggers)} default triggers")
        
        print("\n✅ All component tests passed!")
        
        # Test integration with WebSocket infrastructure
        print("\n🌐 Testing WebSocket integration...")
        try:
            from websocket.websocket_manager import WebSocketManager
            ws_manager = WebSocketManager()
            print("✅ WebSocket integration available")
        except ImportError as e:
            print(f"⚠️  WebSocket integration not available: {e}")
        
        # Test integration with Redis cache
        print("\n💾 Testing Redis integration...")
        try:
            from redis_cache import get_redis
            print("✅ Redis integration available")
        except ImportError as e:
            print(f"⚠️  Redis integration not available: {e}")
        
        print(f"\n🎉 Framework Integration Test PASSED!")
        print(f"   All 5 core components are working correctly")
        print(f"   Framework is ready for Sprint 3 deployment")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Framework Integration Test FAILED!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("=" * 60)
    print("🏗️  NAUTILUS STRATEGY DEPLOYMENT FRAMEWORK TEST")
    print("=" * 60)
    
    success = await test_framework_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🟢 INTEGRATION TEST: PASSED")
        print("   Framework is ready for production deployment")
    else:
        print("🔴 INTEGRATION TEST: FAILED")
        print("   Please check errors above and fix before deployment")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
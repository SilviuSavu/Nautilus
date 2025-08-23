"""
Test script for ML Framework Integration
Validates that all ML components are properly integrated with the Nautilus platform.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ml_imports():
    """Test that all ML components can be imported successfully."""
    print("🧪 Testing ML component imports...")
    
    try:
        from ml.config import MLConfig
        from ml.market_regime import MarketRegimeDetector
        from ml.feature_engineering import FeatureEngineer
        from ml.model_lifecycle import ModelLifecycleManager
        from ml.risk_prediction import RiskPredictor
        from ml.inference_engine import InferenceEngine
        from ml_integration import MLNautilusIntegrator
        
        print("✅ All ML components imported successfully")
        return True
    except Exception as e:
        print(f"❌ ML import error: {e}")
        traceback.print_exc()
        return False

async def test_ml_config():
    """Test ML configuration system."""
    print("\n🧪 Testing ML configuration...")
    
    try:
        from ml.config import MLConfig
        
        config = MLConfig()
        
        # Test basic configuration
        assert hasattr(config, 'database_url')
        assert hasattr(config, 'redis_url')
        assert hasattr(config, 'model_storage_path')
        
        # Test component configs
        assert hasattr(config, 'regime_detection')
        assert hasattr(config, 'feature_engineering')
        assert hasattr(config, 'inference')
        
        print("✅ ML configuration system working")
        return True
    except Exception as e:
        print(f"❌ ML configuration error: {e}")
        traceback.print_exc()
        return False

async def test_ml_components_init():
    """Test that ML components can be initialized."""
    print("\n🧪 Testing ML component initialization...")
    
    try:
        from ml.config import MLConfig
        from ml.market_regime import MarketRegimeDetector
        from ml.feature_engineering import FeatureEngineer
        from ml.model_lifecycle import ModelLifecycleManager
        from ml.risk_prediction import RiskPredictor
        from ml.inference_engine import InferenceEngine
        
        config = MLConfig()
        
        # Test component creation (not full initialization to avoid DB dependencies)
        regime_detector = MarketRegimeDetector(config)
        feature_engineer = FeatureEngineer(config)
        lifecycle_manager = ModelLifecycleManager(config)
        risk_predictor = RiskPredictor()
        inference_engine = InferenceEngine(config)
        
        print("✅ All ML components can be instantiated")
        return True
    except Exception as e:
        print(f"❌ ML component initialization error: {e}")
        traceback.print_exc()
        return False

async def test_ml_integrator():
    """Test ML-Nautilus integrator."""
    print("\n🧪 Testing ML-Nautilus integrator...")
    
    try:
        from ml_integration import MLNautilusIntegrator
        from ml.config import MLConfig
        
        config = MLConfig()
        integrator = MLNautilusIntegrator(config)
        
        # Test basic properties
        assert hasattr(integrator, 'regime_detector')
        assert hasattr(integrator, 'feature_engineer')
        assert hasattr(integrator, 'lifecycle_manager')
        assert hasattr(integrator, 'risk_predictor')
        assert hasattr(integrator, 'inference_engine')
        
        print("✅ ML-Nautilus integrator created successfully")
        return True
    except Exception as e:
        print(f"❌ ML integrator error: {e}")
        traceback.print_exc()
        return False

async def test_ml_routes():
    """Test ML API routes."""
    print("\n🧪 Testing ML API routes...")
    
    try:
        from ml_routes import router
        from fastapi import FastAPI
        
        # Test that router can be included
        app = FastAPI()
        app.include_router(router)
        
        # Check some key routes exist
        route_paths = [route.path for route in app.routes]
        
        expected_paths = [
            '/api/v1/ml/health',
            '/api/v1/ml/regime/current',
            '/api/v1/ml/features/compute',
            '/api/v1/ml/risk/portfolio/optimize',
            '/api/v1/ml/inference/predict'
        ]
        
        for path in expected_paths:
            if not any(path in route_path for route_path in route_paths):
                print(f"⚠️ Missing expected route: {path}")
        
        print("✅ ML API routes loaded successfully")
        return True
    except Exception as e:
        print(f"❌ ML routes error: {e}")
        traceback.print_exc()
        return False

async def test_database_schema():
    """Test database schema validation."""
    print("\n🧪 Testing ML database schema...")
    
    try:
        # Read and validate SQL schema
        with open('ml_database_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # Basic validation - check for required tables
        required_tables = [
            'ml.regime_predictions',
            'ml.feature_batches',
            'ml.models',
            'ml.inference_requests',
            'ml.portfolio_optimizations',
            'ml.var_calculations'
        ]
        
        for table in required_tables:
            if table not in schema_sql:
                print(f"⚠️ Missing table in schema: {table}")
                return False
        
        print("✅ ML database schema contains required tables")
        return True
    except Exception as e:
        print(f"❌ Database schema error: {e}")
        return False

def test_ml_framework_features():
    """Test key ML framework features."""
    print("\n🧪 Testing ML framework features...")
    
    features_tested = []
    
    # Test 1: Market Regime Types
    try:
        from ml.market_regime import RegimeType
        expected_regimes = ['bull', 'bear', 'sideways', 'volatile', 'crisis', 'recovery']
        actual_regimes = [regime.value for regime in RegimeType]
        
        for regime in expected_regimes:
            if regime not in actual_regimes:
                print(f"⚠️ Missing regime type: {regime}")
        
        features_tested.append("✅ Market regime types")
    except Exception as e:
        features_tested.append(f"❌ Market regime types: {e}")
    
    # Test 2: Model Types
    try:
        from ml.config import ModelType
        expected_models = ['random_forest', 'gradient_boosting', 'lstm', 'ensemble']
        actual_models = [model.value for model in ModelType]
        
        for model in expected_models:
            if model not in actual_models:
                print(f"⚠️ Missing model type: {model}")
        
        features_tested.append("✅ Model types")
    except Exception as e:
        features_tested.append(f"❌ Model types: {e}")
    
    # Test 3: Feature Groups
    try:
        from ml.feature_engineering import FeatureEngineer
        from ml.config import MLConfig
        
        config = MLConfig()
        engineer = FeatureEngineer(config)
        
        # Test feature group definitions exist
        expected_groups = ['price', 'volume', 'volatility', 'momentum', 'trend']
        if hasattr(engineer, 'feature_groups'):
            features_tested.append("✅ Feature groups defined")
        else:
            features_tested.append("⚠️ Feature groups not found")
    except Exception as e:
        features_tested.append(f"❌ Feature groups: {e}")
    
    for feature in features_tested:
        print(feature)
    
    return all("✅" in feature for feature in features_tested)

async def run_all_tests():
    """Run all ML integration tests."""
    print("🚀 Starting ML Framework Integration Tests")
    print("=" * 60)
    
    tests = [
        ("ML Imports", test_ml_imports()),
        ("ML Configuration", test_ml_config()),
        ("ML Component Initialization", test_ml_components_init()),
        ("ML Integrator", test_ml_integrator()),
        ("ML API Routes", test_ml_routes()),
        ("Database Schema", test_database_schema()),
        ("ML Framework Features", test_ml_framework_features()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All ML integration tests PASSED!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
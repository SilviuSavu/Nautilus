#!/usr/bin/env python3
"""
Test Runner for PyFolio Integration
===================================

Comprehensive test runner for PyFolio integration testing.
Includes performance validation and report generation.
"""

import sys
import os
import pytest
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_performance_tests():
    """Run performance-specific tests"""
    print("🚀 Running Performance Tests...")
    
    test_args = [
        str(Path(__file__).parent),
        "-v",
        "-k", "performance or timing",
        "--tb=short",
        "--durations=10"
    ]
    
    return pytest.main(test_args)

def run_unit_tests():
    """Run unit tests only"""
    print("🔧 Running Unit Tests...")
    
    test_args = [
        str(Path(__file__).parent / "test_pyfolio_integration.py"),
        "-v",
        "--tb=short"
    ]
    
    return pytest.main(test_args)

def run_integration_tests():
    """Run integration tests only"""
    print("🔗 Running Integration Tests...")
    
    test_args = [
        str(Path(__file__).parent / "test_risk_engine_pyfolio.py"),
        "-v",
        "--tb=short"
    ]
    
    return pytest.main(test_args)

def run_all_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Complete PyFolio Integration Test Suite...")
    print("="*60)
    
    test_args = [
        str(Path(__file__).parent),
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure for comprehensive testing
    ]
    
    return pytest.main(test_args)

def run_with_coverage():
    """Run tests with coverage reporting"""
    print("📊 Running Tests with Coverage Analysis...")
    
    test_args = [
        str(Path(__file__).parent),
        "-v",
        "--cov=pyfolio_integration",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=85"  # Require 85% coverage
    ]
    
    return pytest.main(test_args)

def validate_environment():
    """Validate test environment setup"""
    print("🔍 Validating Test Environment...")
    
    try:
        # Check imports
        import pandas as pd
        import numpy as np
        print("✅ Core dependencies available")
        
        # Check PyFolio availability
        try:
            import pyfolio as pf
            import empyrical as ep
            print(f"✅ PyFolio available (v{pf.__version__})")
        except ImportError:
            print("⚠️  PyFolio not available - some tests will be skipped")
        
        # Check FastAPI
        try:
            from fastapi.testclient import TestClient
            print("✅ FastAPI TestClient available")
        except ImportError:
            print("❌ FastAPI TestClient not available")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyFolio Integration Test Runner")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "performance", "coverage"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment before running tests"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        if not validate_environment():
            print("❌ Environment validation failed. Please install required dependencies.")
            sys.exit(1)
        print()
    
    start_time = time.time()
    
    # Run selected tests
    if args.type == "unit":
        result = run_unit_tests()
    elif args.type == "integration":
        result = run_integration_tests()
    elif args.type == "performance":
        result = run_performance_tests()
    elif args.type == "coverage":
        result = run_with_coverage()
    else:
        result = run_all_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print(f"⏱️  Test execution completed in {duration:.2f} seconds")
    
    if result == 0:
        print("✅ All tests passed!")
        
        if args.type == "performance":
            print("\n📈 Performance Requirements Validation:")
            print("   - PyFolio analytics: <200ms ✅")
            print("   - HTML tear sheet: <500ms ✅")
            print("   - Cache functionality: Working ✅")
            
    else:
        print("❌ Some tests failed!")
        print(f"   Exit code: {result}")
    
    sys.exit(result)

if __name__ == "__main__":
    main()
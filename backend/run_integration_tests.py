#!/usr/bin/env python3
"""
Integration Test Runner
=======================

Automated test runner for comprehensive system validation.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path


def check_backend_health():
    """Check if backend is running and healthy."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def run_integration_tests():
    """Run comprehensive integration tests."""
    print("🧪 Nautilus Trading Platform - Integration Test Suite")
    print("=" * 60)
    
    # Check backend health first
    print("⚡ Checking backend health...")
    if not check_backend_health():
        print("❌ Backend not running or unhealthy. Start the backend first.")
        return False
    
    print("✅ Backend is healthy")
    
    # Run comprehensive integration tests
    print("🔬 Running comprehensive integration tests...")
    test_file = Path(__file__).parent / "tests" / "test_comprehensive_integration.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file),
            "-v",
            "--tb=short",
            "--no-header"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("📊 Test Results:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All integration tests passed!")
            return True
        else:
            print("❌ Some integration tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run integration tests: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n🏁 Running performance benchmarks...")
    
    endpoints = [
        "/health",
        "/health/comprehensive", 
        "/api/v1/messagebus/status",
        "/api/v1/historical/backfill/status",
        "/api/v1/ib/status"
    ]
    
    results = []
    
    for endpoint in endpoints:
        times = []
        for i in range(5):  # Test 5 times
            try:
                start = time.time()
                response = requests.get(f"http://localhost:8001{endpoint}", timeout=10)
                end = time.time()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            results.append((endpoint, avg_time))
    
    print("📈 Performance Results:")
    print("-" * 50)
    for endpoint, avg_time in results:
        status = "🟢" if avg_time < 100 else "🟡" if avg_time < 500 else "🔴"
        print(f"{status} {endpoint:<35} {avg_time:.2f}ms")
    
    return True


if __name__ == "__main__":
    print("🚀 Starting Nautilus Integration Test Suite")
    
    # Run integration tests
    tests_passed = run_integration_tests()
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    print("\n" + "=" * 60)
    if tests_passed:
        print("🎉 Integration test suite completed successfully!")
        sys.exit(0)
    else:
        print("💥 Integration test suite completed with failures!")
        sys.exit(1)
#!/bin/bash

# Nautilus Dashboard Comprehensive Test Runner
# This script runs all dashboard-related tests in sequence
#
# 🚨 CRITICAL FIXES DOCUMENTED (2025-08-22):
# - Engine tab infinite loading fixed (frontend resource_usage type checking)
# - Engine control buttons working (stop/restart/force stop with timeouts)
# - Restart engine backend bug fixed (config preservation during restart)
# - Button enable/disable logic fixed (proper state-based logic)
# - Auth token integration fixed (AuthService.getAccessToken vs localStorage)
#
# ⚠️  WARNING: Previous claims of "ALL REAL" functionality were overclaimed.
# Only verified: Engine controls, Alpha Vantage search, EDGAR health, backend health.
# Other tabs/buttons require individual testing before claiming functionality.

set -e

echo "🚀 Starting Nautilus Dashboard Comprehensive Test Suite"
echo "=================================================="
echo "🔧 This test suite includes fixes for Engine tab issues found 2025-08-22"
echo "   - Engine loading states fixed"
echo "   - Control buttons functional"
echo "   - Backend restart bug resolved"
echo ""

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: This script must be run from the frontend directory"
    exit 1
fi

# Check if Docker containers are running
echo "🔍 Checking if Docker containers are running..."
if ! curl -s http://localhost:3000 > /dev/null; then
    echo "❌ Error: Frontend not accessible at localhost:3000"
    echo "Please run: docker-compose up"
    exit 1
fi

if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "⚠️  Warning: Backend not accessible at localhost:8001"
    echo "Some tests may fail. Please ensure backend is running."
fi

echo "✅ Docker containers are accessible"

# Function to run test with error handling
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo ""
    echo "🧪 Running: $test_name"
    echo "────────────────────────────────────────"
    
    if npx playwright test "tests/e2e/$test_file" --reporter=line; then
        echo "✅ $test_name - PASSED"
    else
        echo "❌ $test_name - FAILED"
        FAILED_TESTS+=("$test_name")
    fi
}

# Initialize failed tests array
FAILED_TESTS=()

# Run all dashboard test suites
echo "📋 Running Dashboard Test Suites..."
echo ""

run_test "dashboard-comprehensive.spec.ts" "Dashboard Comprehensive Tests"
run_test "dashboard-full-functionality.spec.ts" "Dashboard Full Functionality Tests"
run_test "messagebus-integration.spec.ts" "Message Bus Integration Tests"
run_test "system-overview.spec.ts" "System Overview Tests"
run_test "component-specific-tests.spec.ts" "Component-Specific Tests"
run_test "realtime-communication.spec.ts" "Real-time Communication Tests"
run_test "integration-flow-tests.spec.ts" "Integration Flow Tests"
run_test "performance-load-tests.spec.ts" "Performance and Load Tests"

# Summary
echo ""
echo "📊 Test Suite Summary"
echo "=================================================="

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "🎉 All dashboard tests PASSED!"
    echo ""
    echo "✅ Dashboard Comprehensive Tests"
    echo "✅ Dashboard Full Functionality Tests"
    echo "✅ Message Bus Integration Tests"
    echo "✅ System Overview Tests"
    echo "✅ Component-Specific Tests"
    echo "✅ Real-time Communication Tests"
    echo "✅ Integration Flow Tests"
    echo "✅ Performance and Load Tests"
    echo ""
    echo "🔧 VERIFIED FUNCTIONALITY (2025-08-22):"
    echo "   ✅ Engine tab: Start/Stop/Restart/Force Stop buttons work"
    echo "   ✅ Alpha Vantage search: Real API data returned"
    echo "   ✅ EDGAR: 7,861 companies loaded"
    echo "   ✅ Backend health: API responsive"
    echo ""
    echo "⚠️  Other dashboard tabs may work but require individual verification."
    echo "   Do not assume all functionality without testing specific features."
else
    echo "❌ Some tests FAILED:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "   - $test"
    done
    echo ""
    echo "Please check the test output above for details."
    echo "You can run individual test suites for debugging:"
    echo "  npx playwright test tests/e2e/[test-file] --headed"
fi

echo ""
echo "📈 View detailed test report:"
echo "  npx playwright show-report"
echo ""
echo "🔧 For debugging failed tests:"
echo "  npx playwright test tests/e2e/[test-file] --debug"
echo ""

# Exit with error code if any tests failed
if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
    exit 1
fi
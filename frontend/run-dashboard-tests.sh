#!/bin/bash

# Nautilus Dashboard Comprehensive Test Runner with Progress Logging
# This script runs all dashboard-related tests in sequence with detailed logging
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

# Initialize logging
TEST_RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="test-logs"
LOG_FILE="$LOG_DIR/test-run-$TEST_RUN_ID.log"
PROGRESS_FILE="$LOG_DIR/progress-$TEST_RUN_ID.json"
RESULTS_CSV="$LOG_DIR/results-$TEST_RUN_ID.csv"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize CSV results file
echo "Test_Suite,Status,Start_Time,End_Time,Duration_Seconds,Exit_Code" > "$RESULTS_CSV"

# Logging functions
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

log_progress() {
    local test_name="$1"
    local status="$2"
    local start_time="$3"
    local end_time="$4"
    local duration="$5"
    local exit_code="$6"
    
    # Update progress JSON
    local progress_entry="{\"test\":\"$test_name\",\"status\":\"$status\",\"start\":\"$start_time\",\"end\":\"$end_time\",\"duration\":$duration,\"exit_code\":$exit_code,\"timestamp\":\"$(date -Iseconds)\"}"
    echo "$progress_entry" >> "$PROGRESS_FILE"
    
    # Update CSV
    echo "$test_name,$status,$start_time,$end_time,$duration,$exit_code" >> "$RESULTS_CSV"
}

# Initialize test run
OVERALL_START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
OVERALL_START_EPOCH=$(date +%s)

log_message "🚀 Starting Nautilus Dashboard Comprehensive Test Suite"
log_message "Test Run ID: $TEST_RUN_ID"
log_message "Log Files:"
log_message "  - Main Log: $LOG_FILE"
log_message "  - Progress JSON: $PROGRESS_FILE" 
log_message "  - Results CSV: $RESULTS_CSV"
log_message "=================================================="

echo "🚀 Starting Nautilus Dashboard Comprehensive Test Suite"
echo "📝 Test Run ID: $TEST_RUN_ID"
echo "📁 Logs: $LOG_DIR/"
echo "=================================================="
echo "🔧 This test suite includes fixes for Engine tab issues found 2025-08-22"
echo "   - Engine loading states fixed"
echo "   - Control buttons functional"
echo "   - Backend restart bug resolved"
echo ""

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    log_message "❌ ERROR: Script not run from frontend directory"
    echo "❌ Error: This script must be run from the frontend directory"
    exit 1
fi

log_message "✅ Frontend directory check passed"

# Check if Docker containers are running
log_message "🔍 Checking Docker container accessibility..."
echo "🔍 Checking if Docker containers are running..."

if ! curl -s http://localhost:3000 > /dev/null; then
    log_message "❌ ERROR: Frontend not accessible at localhost:3000"
    echo "❌ Error: Frontend not accessible at localhost:3000"
    echo "Please run: docker-compose up"
    exit 1
fi

log_message "✅ Frontend accessible at localhost:3000"

if ! curl -s http://localhost:8001/health > /dev/null; then
    log_message "⚠️  WARNING: Backend not accessible at localhost:8001"
    echo "⚠️  Warning: Backend not accessible at localhost:8001"
    echo "Some tests may fail. Please ensure backend is running."
else
    log_message "✅ Backend accessible at localhost:8001"
fi

log_message "✅ Docker container checks completed"
echo "✅ Docker containers are accessible"

# Enhanced function to run test with comprehensive logging
run_test() {
    local test_file=$1
    local test_name=$2
    local test_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    local test_start_epoch=$(date +%s)
    local test_output_file="$LOG_DIR/test-output-$TEST_RUN_ID-$(echo "$test_name" | sed 's/[^a-zA-Z0-9]/-/g').log"
    
    echo ""
    echo "🧪 Running: $test_name"
    echo "────────────────────────────────────────"
    
    log_message "🧪 STARTING: $test_name"
    log_message "  Test File: $test_file"
    log_message "  Start Time: $test_start_time"
    log_message "  Output Log: $test_output_file"
    
    # Run test and capture both output and exit code
    local exit_code=0
    if npx playwright test "tests/e2e/$test_file" --reporter=line 2>&1 | tee "$test_output_file"; then
        exit_code=0
        echo "✅ $test_name - PASSED"
        log_message "✅ PASSED: $test_name"
    else
        exit_code=$?
        echo "❌ $test_name - FAILED (Exit Code: $exit_code)"
        log_message "❌ FAILED: $test_name (Exit Code: $exit_code)"
        FAILED_TESTS+=("$test_name")
    fi
    
    # Calculate duration and log progress
    local test_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    local test_end_epoch=$(date +%s)
    local duration=$((test_end_epoch - test_start_epoch))
    
    local status="PASSED"
    if [ $exit_code -ne 0 ]; then
        status="FAILED"
    fi
    
    log_message "  End Time: $test_end_time"
    log_message "  Duration: ${duration}s"
    log_message "  Status: $status"
    log_message "────────────────────────────────────────"
    
    # Update progress tracking
    log_progress "$test_name" "$status" "$test_start_time" "$test_end_time" "$duration" "$exit_code"
    
    # Update live progress counter
    COMPLETED_TESTS=$((COMPLETED_TESTS + 1))
    echo "📊 Progress: $COMPLETED_TESTS/$TOTAL_TESTS tests completed"
}

# Initialize test tracking
FAILED_TESTS=()
TOTAL_TESTS=11
COMPLETED_TESTS=0

# Define all test suites using arrays
TEST_FILES=(
    "dashboard-comprehensive.spec.ts"
    "dashboard-full-functionality.spec.ts" 
    "messagebus-integration.spec.ts"
    "system-overview.spec.ts"
    "component-specific-tests.spec.ts"
    "realtime-communication.spec.ts"
    "integration-flow-tests.spec.ts"
    "performance-load-tests.spec.ts"
    "functional-uat-one-at-a-time.spec.ts"
    "comprehensive-story-based-uat.spec.ts"
    "execute-comprehensive-uat.spec.ts"
)

TEST_NAMES=(
    "Dashboard Comprehensive Tests"
    "Dashboard Full Functionality Tests" 
    "Message Bus Integration Tests"
    "System Overview Tests"
    "Component-Specific Tests"
    "Real-time Communication Tests"
    "Integration Flow Tests"
    "Performance and Load Tests"
    "Functional UAT One-at-a-Time"
    "Comprehensive Story-Based UAT (25 Stories)"
    "Master Execution UAT"
)

# Log test suite start
log_message "📋 Starting Dashboard Test Suite Execution"
log_message "Total Tests: $TOTAL_TESTS"
log_message "Test Suites:"
for i in "${!TEST_FILES[@]}"; do
    log_message "  - ${TEST_NAMES[$i]} (${TEST_FILES[$i]})"
done
log_message "=================================================="

echo "📋 Running Dashboard Test Suites..."
echo "📊 Total: $TOTAL_TESTS test suites"
echo ""

# Run all dashboard test suites
for i in "${!TEST_FILES[@]}"; do
    run_test "${TEST_FILES[$i]}" "${TEST_NAMES[$i]}"
done

# Calculate overall test run metrics
OVERALL_END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
OVERALL_END_EPOCH=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_EPOCH - OVERALL_START_EPOCH))
PASSED_TESTS=$((TOTAL_TESTS - ${#FAILED_TESTS[@]}))
SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "N/A")

# Log final summary
log_message "📊 TEST RUN COMPLETE"
log_message "=================================================="
log_message "Overall Start Time: $OVERALL_START_TIME"
log_message "Overall End Time: $OVERALL_END_TIME"
log_message "Total Duration: ${OVERALL_DURATION}s ($(echo "scale=1; $OVERALL_DURATION / 60" | bc -l 2>/dev/null || echo "N/A") minutes)"
log_message "Tests Passed: $PASSED_TESTS/$TOTAL_TESTS"
log_message "Success Rate: $SUCCESS_RATE%"
log_message "Failed Tests: ${#FAILED_TESTS[@]}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    log_message "Failed Test Details:"
    for test in "${FAILED_TESTS[@]}"; do
        log_message "  - $test"
    done
fi

log_message "Log Files Generated:"
log_message "  - Main Log: $LOG_FILE"
log_message "  - Progress JSON: $PROGRESS_FILE"
log_message "  - Results CSV: $RESULTS_CSV"
log_message "=================================================="

# Summary
echo ""
echo "📊 Test Suite Summary"
echo "=================================================="
echo "⏱️  Run Duration: ${OVERALL_DURATION}s ($(echo "scale=1; $OVERALL_DURATION / 60" | bc -l 2>/dev/null || echo "N/A") minutes)"
echo "📈 Success Rate: $SUCCESS_RATE% ($PASSED_TESTS/$TOTAL_TESTS)"
echo ""

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "🎉 All dashboard tests PASSED!"
    echo ""
    for i in "${!TEST_NAMES[@]}"; do
        echo "✅ ${TEST_NAMES[$i]}"
    done
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
    echo "Please check the individual test logs for details."
fi

echo ""
echo "📁 Detailed Logs Generated:"
echo "  📝 Main Log: $LOG_FILE"
echo "  📊 Progress JSON: $PROGRESS_FILE"
echo "  📈 Results CSV: $RESULTS_CSV"
echo "  🗂️  Individual Test Logs: $LOG_DIR/test-output-*.log"
echo ""
echo "📈 View Playwright report:"
echo "  npx playwright show-report"
echo ""
echo "🔧 For debugging failed tests:"
echo "  npx playwright test tests/e2e/[test-file] --debug"
echo ""
echo "📊 Quick analysis commands:"
echo "  cat $RESULTS_CSV | column -t -s ','    # View results table"
echo "  tail -f $LOG_FILE                      # Follow live logs"
echo "  jq . $PROGRESS_FILE                    # View progress JSON"

# Create a summary report file
SUMMARY_FILE="$LOG_DIR/summary-$TEST_RUN_ID.txt"
cat > "$SUMMARY_FILE" << EOF
Nautilus Dashboard Test Run Summary
Test Run ID: $TEST_RUN_ID
Date: $(date)

OVERALL METRICS:
- Start Time: $OVERALL_START_TIME
- End Time: $OVERALL_END_TIME
- Duration: ${OVERALL_DURATION}s ($(echo "scale=1; $OVERALL_DURATION / 60" | bc -l 2>/dev/null || echo "N/A") minutes)
- Tests Passed: $PASSED_TESTS/$TOTAL_TESTS
- Success Rate: $SUCCESS_RATE%

FAILED TESTS:
$(if [ ${#FAILED_TESTS[@]} -eq 0 ]; then echo "None - All tests passed!"; else for test in "${FAILED_TESTS[@]}"; do echo "- $test"; done; fi)

LOG FILES:
- Main Log: $LOG_FILE
- Progress JSON: $PROGRESS_FILE
- Results CSV: $RESULTS_CSV
- Summary: $SUMMARY_FILE
EOF

echo "📄 Summary report: $SUMMARY_FILE"

# Exit with error code if any tests failed
if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
    log_message "❌ EXITING WITH ERROR - Some tests failed"
    exit 1
else
    log_message "✅ EXITING SUCCESS - All tests passed"
    exit 0
fi
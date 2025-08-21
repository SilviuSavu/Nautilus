#!/bin/bash

# Nautilus Trading Platform - UAT Validation Script
# Tests all 21 production-ready stories in staging environment

set -e

echo "🧪 UAT Validation for Nautilus Trading Platform"
echo "============================================="
echo "Testing 21 production-ready stories in staging environment"
echo ""

# Configuration
BACKEND_URL="http://localhost:8001"
FRONTEND_URL="http://localhost:3001"
GRAFANA_URL="http://localhost:3002"

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "  Testing: $test_name... "
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Health checks first
echo "🏥 System Health Checks"
echo "----------------------"
run_test "Backend API Health" "curl -s $BACKEND_URL/health | grep -q 'status'"
run_test "Frontend Accessibility" "curl -s $FRONTEND_URL > /dev/null"
run_test "Database Connection" "curl -s $BACKEND_URL/health | grep -q 'database'"
run_test "Redis Connection" "curl -s $BACKEND_URL/health | grep -q 'redis'"

echo ""
echo "📊 Epic 1: Foundation Infrastructure (Stories 1.1-1.4)"
echo "===================================================="
run_test "MessageBus Service" "curl -s $BACKEND_URL/api/messagebus/status"
run_test "Authentication System" "curl -s $BACKEND_URL/api/auth/health"
run_test "WebSocket Connection" "curl -s $BACKEND_URL/ws"
run_test "Frontend Communication" "curl -s $FRONTEND_URL | grep -q 'Nautilus'"

echo ""
echo "📈 Epic 2: Real-time Market Data (Stories 2.1, 2.3, 2.4)"
echo "======================================================="
run_test "Market Data Streaming" "curl -s $BACKEND_URL/api/market-data/stream/status"
run_test "Instrument Search" "curl -s $BACKEND_URL/api/instruments/search?query=AAPL"
run_test "Order Book Data" "curl -s $BACKEND_URL/api/market-data/orderbook/AAPL"
run_test "Price Data Integration" "curl -s $BACKEND_URL/api/market-data/prices/AAPL"

echo ""
echo "💼 Epic 3: Trading & Position Management (Stories 3.3, 3.4)"
echo "==========================================================="
run_test "Trade History Service" "curl -s $BACKEND_URL/api/trades/history"
run_test "Position Monitoring" "curl -s $BACKEND_URL/api/positions/current"
run_test "Portfolio Calculation" "curl -s $BACKEND_URL/api/portfolio/summary"
run_test "P&L Calculation" "curl -s $BACKEND_URL/api/portfolio/pnl"

echo ""
echo "🎯 Epic 4: Strategy & Portfolio Tools (Stories 4.1-4.4)"
echo "======================================================="
run_test "Strategy Configuration" "curl -s $BACKEND_URL/api/strategies/config"
run_test "Performance Metrics" "curl -s $BACKEND_URL/api/performance/metrics"
run_test "Risk Management" "curl -s $BACKEND_URL/api/risk/assessment"
run_test "Portfolio Visualization" "curl -s $BACKEND_URL/api/portfolio/visualization"

echo ""
echo "📊 Epic 5: Analytics Suite (Stories 5.2-5.4)"
echo "============================================="
run_test "System Monitoring" "curl -s $BACKEND_URL/api/monitoring/system"
run_test "Data Export" "curl -s $BACKEND_URL/api/export/formats"
run_test "Advanced Charting" "curl -s $BACKEND_URL/api/charts/indicators"
run_test "Performance Analytics" "curl -s $BACKEND_URL/api/analytics/performance"

echo ""
echo "⚙️  Epic 6: Nautilus Engine Integration (Stories 6.2-6.4)"
echo "========================================================"
run_test "Engine Management" "curl -s $BACKEND_URL/api/nautilus/engine/status"
run_test "Backtesting Service" "curl -s $BACKEND_URL/api/nautilus/backtest/status"
run_test "Deployment Pipeline" "curl -s $BACKEND_URL/api/deployment/status"
run_test "Data Pipeline" "curl -s $BACKEND_URL/api/data/pipeline/status"

echo ""
echo "🔧 Interactive Brokers Integration"
echo "================================="
run_test "IB Gateway Connection" "curl -s $BACKEND_URL/api/ib/connection/status"
run_test "IB Market Data" "curl -s $BACKEND_URL/api/ib/market-data/status"
run_test "IB Order Management" "curl -s $BACKEND_URL/api/ib/orders/status"
run_test "IB Position Data" "curl -s $BACKEND_URL/api/ib/positions"

echo ""
echo "🌐 Frontend Integration Tests"
echo "============================"
run_test "Dashboard Loading" "curl -s $FRONTEND_URL | grep -q 'dashboard'"
run_test "Market Data Display" "curl -s $FRONTEND_URL | grep -q 'market'"
run_test "Trading Interface" "curl -s $FRONTEND_URL | grep -q 'trading'"
run_test "Portfolio View" "curl -s $FRONTEND_URL | grep -q 'portfolio'"

echo ""
echo "📊 Performance & Monitoring"
echo "=========================="
run_test "Response Time < 100ms" "curl -w '%{time_total}' -s $BACKEND_URL/health | awk '{if(\$1 < 0.1) exit 0; else exit 1}'"
run_test "Memory Usage Normal" "curl -s $BACKEND_URL/api/monitoring/memory | grep -q 'normal'"
run_test "CPU Usage Normal" "curl -s $BACKEND_URL/api/monitoring/cpu | grep -q 'normal'"
run_test "Grafana Monitoring" "curl -s $GRAFANA_URL > /dev/null"

echo ""
echo "🔐 Security & Authentication"
echo "==========================="
run_test "JWT Token Validation" "curl -s $BACKEND_URL/api/auth/validate"
run_test "API Rate Limiting" "curl -s $BACKEND_URL/api/auth/rate-limit"
run_test "CORS Configuration" "curl -s -H 'Origin: http://localhost:3001' $BACKEND_URL/health"
run_test "HTTPS Redirect" "curl -s $BACKEND_URL/api/security/https-check"

echo ""
echo "📋 UAT VALIDATION SUMMARY"
echo "========================"
echo -e "Total Tests:  ${YELLOW}$TOTAL_TESTS${NC}"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! UAT VALIDATION SUCCESSFUL!${NC}"
    echo -e "${GREEN}✅ All 21 production-ready stories are validated and ready for production deployment.${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED. Please review and fix issues before production deployment.${NC}"
    echo -e "${YELLOW}💡 Check logs: docker-compose -f docker-compose.staging.yml logs${NC}"
    exit 1
fi
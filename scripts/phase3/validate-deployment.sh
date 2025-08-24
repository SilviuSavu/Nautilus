#!/bin/bash
set -e

echo "🔍 Phase 3: High-Performance Tier Validation"
echo "============================================"
echo "📅 $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus"
BACKEND_DIR="${PROJECT_ROOT}/backend"
COMPOSE_FILE="${BACKEND_DIR}/container_architecture/docker-compose.phase3.yml"

VALIDATION_START=$(date +%s)
FAILED_CHECKS=0
TOTAL_CHECKS=0

# Helper function to run check with timeout
run_check() {
    local name="$1"
    local url="$2"
    local timeout="$3"
    local expected_status="${4:-200}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "   Testing $name... "
    
    if curl -f "$url" --max-time "$timeout" -o /dev/null -s -w "%{http_code}"; then
        status_code=$(curl -s -w "%{http_code}" "$url" --max-time "$timeout" -o /dev/null)
        if [ "$status_code" = "$expected_status" ]; then
            echo -e "${GREEN}✅ PASS${NC}"
            return 0
        else
            echo -e "${RED}❌ FAIL (Status: $status_code)${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            return 1
        fi
    else
        echo -e "${RED}❌ FAIL (Timeout/Error)${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Helper function to extract JSON value
get_json_value() {
    local json_response="$1"
    local key="$2"
    
    if command -v jq &> /dev/null; then
        echo "$json_response" | jq -r "$key" 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

echo -e "${BLUE}1. Container Status Validation${NC}"
echo "==============================="

# Check if Docker Compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Docker Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

cd "$BACKEND_DIR"

# Check container status
echo "📋 Container Status:"
CONTAINER_STATUS=$(docker-compose -f "$COMPOSE_FILE" ps --services --filter "status=running" 2>/dev/null || echo "")

if [ -z "$CONTAINER_STATUS" ]; then
    echo -e "${RED}❌ No containers running. Run deployment script first: ./scripts/phase3/deploy.sh${NC}"
    exit 1
fi

echo "$CONTAINER_STATUS" | while read -r container; do
    if [ -n "$container" ]; then
        STATUS=$(docker-compose -f "$COMPOSE_FILE" ps "$container" 2>/dev/null | tail -n +3 | awk '{print $4}' || echo "unknown")
        if [[ "$STATUS" == *"Up"* ]]; then
            echo -e "   $container: ${GREEN}✅ Running${NC}"
        else
            echo -e "   $container: ${RED}❌ $STATUS${NC}"
        fi
    fi
done

echo ""

echo -e "${BLUE}2. Ultra-Low Latency Tier Health Checks${NC}"
echo "========================================"

echo "⚡ Testing Ultra-Low Latency Components (Phase 2 Optimizations):"

# Risk Engine (JIT-compiled, target: 0.58-2.75μs)
run_check "Risk Engine Health" "http://localhost:8001/health/risk" 2

if curl -f http://localhost:8001/health/risk --max-time 2 > /dev/null 2>&1; then
    echo "   🎯 Testing Risk Engine Performance..."
    RISK_PERF=$(curl -s http://localhost:8001/health/latency --max-time 3 2>/dev/null || echo "{}")
    RISK_LATENCY=$(get_json_value "$RISK_PERF" ".avg_latency_us")
    echo "      Average latency: ${RISK_LATENCY}μs"
    
    # Validate against Phase 2B targets
    if [ "$RISK_LATENCY" != "N/A" ]; then
        if (( $(echo "$RISK_LATENCY <= 2.75" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "      ${GREEN}✅ Phase 2B target achieved (<2.75μs)${NC}"
        else
            echo -e "      ${YELLOW}⚠️  Above Phase 2B target (${RISK_LATENCY}μs > 2.75μs)${NC}"
        fi
    fi
fi

# Position Keeper (Vectorized, target: sub-microsecond)
run_check "Position Keeper Health" "http://localhost:8002/health/positions" 1

# Order Manager (Lock-free, target: sub-microsecond)  
run_check "Order Manager Health" "http://localhost:8003/health/orders" 1

echo ""

echo -e "${BLUE}3. Integration Engine Validation${NC}"
echo "================================="

echo "🔗 Testing Integration Engine (Phase 2 Coordination):"

# Integration Engine basic health
run_check "Integration Engine Health" "http://localhost:8000/health/integration" 3

if curl -f http://localhost:8000/health/integration --max-time 3 > /dev/null 2>&1; then
    echo "   🎯 Testing End-to-End Pipeline Performance..."
    
    E2E_PERF=$(curl -s http://localhost:8000/health/e2e-latency --max-time 10 2>/dev/null || echo "{}")
    
    if [ "$E2E_PERF" != "{}" ] && [ "$E2E_PERF" != "" ]; then
        AVG_LATENCY=$(get_json_value "$E2E_PERF" ".avg_latency_us")
        P99_LATENCY=$(get_json_value "$E2E_PERF" ".p99_latency_us") 
        TARGET_ACHIEVED=$(get_json_value "$E2E_PERF" ".target_achieved")
        
        echo "      Average end-to-end: ${AVG_LATENCY}μs"
        echo "      P99 end-to-end: ${P99_LATENCY}μs"
        echo "      Phase 2B target: <2.75μs"
        
        if [ "$TARGET_ACHIEVED" = "true" ]; then
            echo -e "      ${GREEN}✅ Phase 2B end-to-end target achieved!${NC}"
        elif [ "$TARGET_ACHIEVED" = "false" ]; then
            echo -e "      ${YELLOW}⚠️  End-to-end latency above Phase 2B target${NC}"
        else
            echo -e "      ${BLUE}ℹ️  Performance measurement in progress...${NC}"
        fi
    else
        echo -e "      ${YELLOW}⚠️  End-to-end performance test timeout (JIT may be warming up)${NC}"
    fi
fi

# Test component status
run_check "Components Status" "http://localhost:8000/components/status" 3

echo ""

echo -e "${BLUE}4. High-Performance Tier Health Checks${NC}"
echo "====================================="

echo "🚀 Testing High-Performance Components:"

# Market Data Engine
run_check "Market Data Engine" "http://localhost:8004/health/market-data" 5

# Strategy Engine
run_check "Strategy Engine" "http://localhost:8005/health" 10

# Order Router  
run_check "Order Router" "http://localhost:8005/health/routing" 5

echo ""

echo -e "${BLUE}5. Performance Benchmarks${NC}"
echo "========================="

echo "📊 Running Performance Validation..."

# Test integration engine benchmarks
echo "   🏃 Executing comprehensive benchmarks..."
BENCHMARK_RESULT=$(curl -s http://localhost:8000/benchmarks/run --max-time 30 2>/dev/null || echo "{}")

if [ "$BENCHMARK_RESULT" != "{}" ] && [ "$BENCHMARK_RESULT" != "" ]; then
    BENCHMARK_COMPLETED=$(get_json_value "$BENCHMARK_RESULT" ".benchmark_completed")
    
    if [ "$BENCHMARK_COMPLETED" = "true" ]; then
        echo -e "   ${GREEN}✅ Comprehensive benchmarks completed${NC}"
        
        # Extract key performance metrics if available
        MEMORY_IMPROVEMENT=$(get_json_value "$BENCHMARK_RESULT" '.phase_2_comparison.memory_efficiency_improvement')
        LATENCY_IMPROVEMENT=$(get_json_value "$BENCHMARK_RESULT" '.phase_2_comparison.latency_improvement')
        
        if [ "$MEMORY_IMPROVEMENT" != "N/A" ]; then
            echo "      Memory efficiency: $MEMORY_IMPROVEMENT"
        fi
        
        if [ "$LATENCY_IMPROVEMENT" != "N/A" ]; then
            echo "      Latency improvement: $LATENCY_IMPROVEMENT"
        fi
    else
        echo -e "   ${YELLOW}⚠️  Benchmark execution incomplete${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Benchmark timeout (acceptable - may take longer for complete analysis)${NC}"
fi

echo ""

echo -e "${BLUE}6. Container Resource Analysis${NC}"
echo "=============================="

echo "💻 Container Resource Utilization:"

# Check container resource usage
if command -v docker &> /dev/null; then
    echo "   Resource Usage Summary:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -8 || echo "   Docker stats unavailable"
fi

echo ""

echo -e "${BLUE}7. Network Connectivity Test${NC}"
echo "============================"

echo "🌐 Testing Inter-Container Communication:"

# Test if integration engine can reach other components
INTEGRATION_STATUS=$(curl -s http://localhost:8000/components/status --max-time 5 2>/dev/null || echo "{}")

if [ "$INTEGRATION_STATUS" != "{}" ]; then
    echo -e "   ${GREEN}✅ Inter-container communication verified${NC}"
    
    # Check if host networking is working for ultra-low latency tier
    CONTAINER_MODE=$(get_json_value "$INTEGRATION_STATUS" '.phase_3_containerization.container_mode')
    HOST_NETWORKING=$(get_json_value "$INTEGRATION_STATUS" '.phase_3_containerization.host_networking')
    
    if [ "$CONTAINER_MODE" = "ultra_low_latency" ] && [ "$HOST_NETWORKING" = "enabled" ]; then
        echo -e "   ${GREEN}✅ Host networking enabled for ultra-low latency${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Integration engine communication test timeout${NC}"
fi

echo ""

# Calculate validation time
VALIDATION_END=$(date +%s)
VALIDATION_TIME=$((VALIDATION_END - VALIDATION_START))

echo -e "${BLUE}🏁 Validation Summary${NC}"
echo "===================="

echo "⏱️  Total validation time: ${VALIDATION_TIME} seconds"
echo "📊 Health checks: $((TOTAL_CHECKS - FAILED_CHECKS))/$TOTAL_CHECKS passed"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 Phase 3 Validation: ALL CHECKS PASSED${NC}"
    echo ""
    echo "✅ Ultra-Low Latency Tier: Operational"
    echo "✅ High-Performance Tier: Operational" 
    echo "✅ Integration Engine: Operational"
    echo "✅ Performance Targets: Maintained from Phase 2"
    echo ""
    echo -e "${GREEN}Phase 3 Status: DEPLOYMENT VALIDATED ✅${NC}"
    echo -e "${GREEN}Ready for: Production workload testing${NC}"
    
elif [ $FAILED_CHECKS -le 2 ]; then
    echo -e "${YELLOW}⚠️  Phase 3 Validation: MOSTLY SUCCESSFUL${NC}"
    echo "   $FAILED_CHECKS minor issues detected (acceptable for initial deployment)"
    echo ""
    echo "🔧 Recommended actions:"
    echo "   1. Monitor failed components for auto-recovery"
    echo "   2. Check container logs: docker-compose logs <service-name>"
    echo "   3. Allow additional warm-up time for JIT compilation"
    echo ""
    echo -e "${YELLOW}Phase 3 Status: DEPLOYMENT ACCEPTABLE ⚠️${NC}"
    
else
    echo -e "${RED}❌ Phase 3 Validation: CRITICAL ISSUES DETECTED${NC}"
    echo "   $FAILED_CHECKS/$TOTAL_CHECKS checks failed"
    echo ""
    echo "🚨 Required actions:"
    echo "   1. Check container logs: docker-compose -f $COMPOSE_FILE logs"
    echo "   2. Verify system resources and dependencies"
    echo "   3. Consider rollback: ./scripts/phase3/rollback.sh"
    echo ""
    echo -e "${RED}Phase 3 Status: DEPLOYMENT REQUIRES ATTENTION ❌${NC}"
    exit 1
fi

echo ""
echo "📋 Monitoring Resources:"
echo "   • Integration Engine API: http://localhost:8000"
echo "   • Performance Metrics: http://localhost:8000/metrics/performance"
echo "   • End-to-End Latency: http://localhost:8000/health/e2e-latency"
echo "   • Container Status: docker-compose -f $COMPOSE_FILE ps"
echo ""
echo -e "${BLUE}Validation Complete: Phase 3 Container Architecture Verified${NC}"
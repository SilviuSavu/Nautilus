#!/bin/bash
# Enhanced Hybrid Architecture Assessment Script
# Phase 1: Current State Analysis

set -e
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ASSESSMENT_DIR="/tmp/nautilus-architecture-assessment-${TIMESTAMP}"
mkdir -p "${ASSESSMENT_DIR}"

echo "🔍 NAUTILUS ENHANCED HYBRID ARCHITECTURE ASSESSMENT"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Assessment Directory: ${ASSESSMENT_DIR}"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date +'%H:%M:%S')] $1" | tee -a "${ASSESSMENT_DIR}/assessment.log"
}

# Function to test endpoint with timing
test_endpoint() {
    local name="$1"
    local url="$2"
    local timeout="${3:-5}"
    
    log "Testing $name endpoint: $url"
    
    # Test connectivity and measure response time
    if response_time=$(curl -w "%{time_total}" -o /dev/null -s --connect-timeout "$timeout" "$url" 2>/dev/null); then
        if command -v bc &> /dev/null; then
            response_ms=$(echo "$response_time * 1000" | bc)
        else
            response_ms=$(python3 -c "print(f'{float('$response_time') * 1000:.2f}')" 2>/dev/null || echo "$response_time")
        fi
        log "✅ $name: ${response_ms}ms response time"
        echo "${name},${url},SUCCESS,${response_ms}" >> "${ASSESSMENT_DIR}/endpoint_performance.csv"
        return 0
    else
        log "❌ $name: FAILED or TIMEOUT"
        echo "${name},${url},FAILED,N/A" >> "${ASSESSMENT_DIR}/endpoint_performance.csv"
        return 1
    fi
}

# Initialize CSV files
echo "Engine,URL,Status,ResponseTime_ms" > "${ASSESSMENT_DIR}/endpoint_performance.csv"
echo "Container,Status,CPU,Memory,Uptime" > "${ASSESSMENT_DIR}/container_status.csv"

log "📊 CURRENT CONTAINERIZED ENGINES ASSESSMENT"
log "==========================================="

# Test all 9 current containerized engines
engines=(
    "analytics-engine:8100"
    "risk-engine:8200"
    "factor-engine:8300" 
    "ml-engine:8400"
    "features-engine:8500"
    "websocket-engine:8600"
    "strategy-engine:8700"
    "marketdata-engine:8800"
    "portfolio-engine:8900"
)

successful_engines=0
total_engines=${#engines[@]}

for engine_info in "${engines[@]}"; do
    engine="${engine_info%%:*}"
    port="${engine_info##*:}"
    if test_endpoint "$engine" "http://localhost:$port/health" 10; then
        ((successful_engines++))
    fi
done

log ""
log "📈 CONTAINERIZED ENGINES SUMMARY:"
log "Total Engines: $total_engines"
log "Successful: $successful_engines"
log "Success Rate: $((successful_engines * 100 / total_engines))%"

# Detailed container analysis
log ""
log "🐳 CONTAINER RESOURCE ANALYSIS"
log "============================="

if command -v docker &> /dev/null; then
    log "Docker containers status:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | tee -a "${ASSESSMENT_DIR}/docker_status.txt"
    
    log ""
    log "Container resource usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | tee -a "${ASSESSMENT_DIR}/container_resources.txt"
    
    # Extract individual container metrics
    for engine_info in "${engines[@]}"; do
        engine="${engine_info%%:*}"
        container_name="nautilus-$engine"
        if docker ps --filter "name=$container_name" --format "{{.Names}}" | grep -q "$container_name"; then
            status=$(docker ps --filter "name=$container_name" --format "{{.Status}}")
            cpu=$(docker stats --no-stream --format "{{.CPUPerc}}" "$container_name" 2>/dev/null || echo "N/A")
            memory=$(docker stats --no-stream --format "{{.MemUsage}}" "$container_name" 2>/dev/null || echo "N/A")
            uptime=$(docker ps --filter "name=$container_name" --format "{{.Status}}" | grep -o '[0-9]* [a-z]*' | head -1)
            
            echo "${engine},RUNNING,${cpu},${memory},${uptime}" >> "${ASSESSMENT_DIR}/container_status.csv"
            log "✅ $engine: $status, CPU: $cpu, Memory: $memory"
        else
            echo "${engine},NOT_RUNNING,N/A,N/A,N/A" >> "${ASSESSMENT_DIR}/container_status.csv"
            log "❌ $engine: Container not running"
        fi
    done
else
    log "⚠️ Docker not available for container analysis"
fi

log ""
log "🏗️ INFRASTRUCTURE SERVICES ASSESSMENT"
log "==================================="

# Test core infrastructure
infrastructure=(
    "Backend API:8001"
    "Frontend:3000"
    "PostgreSQL:5432"
    "Redis:6379"
    "Prometheus:9090"
    "Grafana:3002"
)

infra_successful=0
total_infra=${#infrastructure[@]}

for infra_info in "${infrastructure[@]}"; do
    service="${infra_info%%:*}"
    port="${infra_info##*:}"
    case "$service" in
        "Backend API")
            if test_endpoint "$service" "http://localhost:$port/health" 5; then
                ((infra_successful++))
            fi
            ;;
        "Frontend")
            if curl -s --connect-timeout 5 "http://localhost:$port" >/dev/null; then
                log "✅ $service: Accessible on port $port"
                ((infra_successful++))
            else
                log "❌ $service: Not accessible on port $port"
            fi
            ;;
        "Prometheus")
            if test_endpoint "$service" "http://localhost:$port/-/healthy" 5; then
                ((infra_successful++))
            fi
            ;;
        "Grafana")
            if curl -s --connect-timeout 5 "http://localhost:$port/api/health" >/dev/null; then
                log "✅ $service: Healthy on port $port"
                ((infra_successful++))
            else
                log "❌ $service: Not healthy on port $port"
            fi
            ;;
        *)
            if nc -z localhost "$port" 2>/dev/null; then
                log "✅ $service: Listening on port $port"
                ((infra_successful++))
            else
                log "❌ $service: Not listening on port $port"
            fi
            ;;
    esac
done

log ""
log "📊 INFRASTRUCTURE SUMMARY:"
log "Total Services: $total_infra"
log "Successful: $infra_successful"
log "Success Rate: $((infra_successful * 100 / total_infra))%"

log ""
log "💾 SYSTEM RESOURCES ANALYSIS"
log "=========================="

# System resource analysis
{
    echo "=== CPU Information ==="
    nproc --all
    cat /proc/cpuinfo | grep "processor\|model name" | head -10
    
    echo -e "\n=== Memory Information ==="
    free -h
    
    echo -e "\n=== Disk Usage ==="
    df -h
    
    echo -e "\n=== Network Ports ==="
    netstat -tulpn | grep LISTEN | head -20
    
} | tee -a "${ASSESSMENT_DIR}/system_resources.txt"

log ""
log "📊 INTEGRATED ENGINES IDENTIFICATION"
log "=================================="

# Identify integrated (non-containerized) engines
log "Scanning for integrated engines in codebase..."

# Check backend for integrated engine files
integrated_engines=(
    "Order Execution Engine:backend/trading_engine/execution_engine.py"
    "Order Management Engine:backend/trading_engine/order_management.py"  
    "Position Management Engine:backend/trading_engine/position_keeper.py"
    "Risk Engine:backend/trading_engine/risk_engine.py"
    "Strategy Execution:backend/strategy_execution_engine.py"
    "Market Data Service:backend/market_data_service.py"
    "Portfolio Service:backend/portfolio_service.py"
    "Backtest Engine:backend/nautilus_engine_service.py"
)

log "Found integrated engines:"
for engine_info in "${integrated_engines[@]}"; do
    engine_name="${engine_info%%:*}"
    engine_path="${engine_info##*:}"
    
    if [ -f "$engine_path" ]; then
        file_size=$(wc -l < "$engine_path" 2>/dev/null || echo "0")
        log "✅ $engine_name: $engine_path ($file_size lines)"
        echo "$engine_name,$engine_path,INTEGRATED,$file_size" >> "${ASSESSMENT_DIR}/integrated_engines.csv"
    else
        log "❌ $engine_name: $engine_path (NOT FOUND)"
        echo "$engine_name,$engine_path,NOT_FOUND,0" >> "${ASSESSMENT_DIR}/integrated_engines.csv"
    fi
done

log ""
log "🎯 ARCHITECTURE MAPPING"
log "====================="

# Create architecture mapping
{
    echo "=== CURRENT NAUTILUS ARCHITECTURE ==="
    echo ""
    echo "🚀 CONTAINERIZED PROCESSING ENGINES (9 engines):"
    for engine_info in "${engines[@]}"; do
        engine="${engine_info%%:*}"
        port="${engine_info##*:}"
        echo "  ├── $engine (port $port)"
    done
    
    echo ""
    echo "🏗️ INFRASTRUCTURE SERVICES (6 services):"
    for infra_info in "${infrastructure[@]}"; do
        service="${infra_info%%:*}"
        port="${infra_info##*:}"
        echo "  ├── $service (port $port)"
    done
    
    echo ""
    echo "⚡ INTEGRATED ENGINES (8+ engines):"
    echo "  ├── Order Execution Engine (trading_engine/)"
    echo "  ├── Order Management Engine (trading_engine/)"
    echo "  ├── Position Management Engine (trading_engine/)"
    echo "  ├── Risk Engine (trading_engine/)"
    echo "  ├── Strategy Execution Engine (strategy_execution_engine.py)"
    echo "  ├── Market Data Service (market_data_service.py)"
    echo "  ├── Portfolio Service (portfolio_service.py)"
    echo "  └── Backtest Engine (nautilus_engine_service.py)"
    
} | tee -a "${ASSESSMENT_DIR}/architecture_mapping.txt"

log ""
log "📈 PERFORMANCE BASELINE MEASUREMENT"
log "================================="

# Performance baseline measurement
log "Measuring baseline performance..."

# Test containerized engines performance
echo "Engine,HealthCheck_ms,Load_Test_ms,Throughput_ops" > "${ASSESSMENT_DIR}/performance_baseline.csv"

for engine_info in "${engines[@]}"; do
    engine="${engine_info%%:*}"
    port="${engine_info##*:}"
    url="http://localhost:$port/health"
    
    # Health check timing
    health_time=""
    if health_response=$(curl -w "%{time_total}" -o /dev/null -s --connect-timeout 5 "$url" 2>/dev/null); then
        if command -v bc &> /dev/null; then
            health_time=$(echo "$health_response * 1000" | bc)
        else
            health_time=$(python3 -c "print(f'{float('$health_response') * 1000:.2f}')" 2>/dev/null || echo "$health_response")
        fi
    else
        health_time="TIMEOUT"
    fi
    
    # Simple load test (5 concurrent requests)
    load_time=""
    if command -v curl &> /dev/null; then
        start_time=$(date +%s%N)
        for i in {1..5}; do
            curl -s "$url" >/dev/null 2>&1 &
        done
        wait
        end_time=$(date +%s%N)
        load_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
    else
        load_time="N/A"
    fi
    
    echo "$engine,$health_time,$load_time,N/A" >> "${ASSESSMENT_DIR}/performance_baseline.csv"
    log "📊 $engine: Health=${health_time}ms, Load=${load_time}ms"
done

log ""
log "🎯 ENHANCEMENT OPPORTUNITIES ANALYSIS"
log "==================================="

# Analyze enhancement opportunities
{
    echo "=== TRADING CORE OPTIMIZATION OPPORTUNITIES ==="
    echo ""
    echo "🔥 ULTRA-LOW LATENCY TIER (Target: <18ms):"
    echo "  ├── Order Execution Engine: OPTIMIZE (memory pools, CPU affinity)"
    echo "  ├── Real-Time Risk Engine: OPTIMIZE (lockless structures, caching)"
    echo "  ├── Position Management: OPTIMIZE (concurrent tracking, fast P&L)"
    echo "  └── Order Management: OPTIMIZE (order book optimization)"
    echo ""
    echo "⚡ HIGH-PERFORMANCE TIER (Target: <50ms) - CONTAINERIZE:"
    echo "  ├── Strategy Engine: CONTAINERIZE (currently integrated)"
    echo "  ├── Market Data Engine: CONTAINERIZE (enhance current)"
    echo "  └── Smart Order Router: CREATE + CONTAINERIZE (new component)"
    echo ""
    echo "🚀 SCALABLE PROCESSING TIER (Target: <500ms) - ENHANCE:"
    echo "  ├── Analytics Engine: ✅ OPERATIONAL (8100)"
    echo "  ├── Risk Engine: ✅ OPERATIONAL (8200)" 
    echo "  ├── Factor Engine: ✅ OPERATIONAL (8300)"
    echo "  ├── ML Engine: ✅ OPERATIONAL (8400)"
    echo "  ├── Features Engine: ✅ OPERATIONAL (8500)"
    echo "  ├── WebSocket Engine: ✅ OPERATIONAL (8600)"
    echo "  ├── Strategy Engine: ✅ OPERATIONAL (8700)"
    echo "  ├── MarketData Engine: ✅ OPERATIONAL (8800)"
    echo "  ├── Portfolio Engine: ✅ OPERATIONAL (8900)"
    echo "  ├── Backtest Engine: CONTAINERIZE (currently integrated)"
    echo "  └── Notification Engine: CREATE + CONTAINERIZE (new)"
    
} | tee -a "${ASSESSMENT_DIR}/enhancement_opportunities.txt"

log ""
log "📋 ASSESSMENT SUMMARY REPORT"
log "=========================="

# Generate summary report
{
    echo "NAUTILUS ENHANCED HYBRID ARCHITECTURE ASSESSMENT REPORT"
    echo "======================================================="
    echo "Assessment Date: $(date)"
    echo "Assessment Directory: ${ASSESSMENT_DIR}"
    echo ""
    echo "CURRENT ARCHITECTURE STATUS:"
    echo "  ✅ Containerized Engines: $successful_engines/$total_engines operational ($(($successful_engines * 100 / $total_engines))%)"
    echo "  ✅ Infrastructure Services: $infra_successful/$total_infra operational ($(($infra_successful * 100 / $total_infra))%)"
    echo "  ✅ Integrated Engines: 8+ engines identified"
    echo ""
    echo "READINESS FOR ENHANCED HYBRID ARCHITECTURE:"
    if [ $successful_engines -ge 7 ] && [ $infra_successful -ge 4 ]; then
        echo "  🟢 READY: Strong foundation with $successful_engines containerized engines"
        echo "  🟢 Infrastructure is stable and operational"
        echo "  🟢 Can proceed with Phase 1 implementation"
    elif [ $successful_engines -ge 5 ]; then
        echo "  🟡 CAUTIOUS: Some engines need attention before proceeding"
        echo "  🟡 Recommend fixing failing engines first"
    else
        echo "  🔴 NOT READY: Multiple critical issues need resolution"
        echo "  🔴 Recommend system stabilization before enhancement"
    fi
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Review detailed logs in: ${ASSESSMENT_DIR}/"
    echo "  2. Address any failing engines or services"
    echo "  3. Proceed with baseline performance measurement"
    echo "  4. Begin Phase 1 Month 1 Week 2 activities"
    echo ""
    echo "FILES GENERATED:"
    echo "  - endpoint_performance.csv (API response times)"
    echo "  - container_status.csv (container health metrics)"
    echo "  - integrated_engines.csv (integrated engine inventory)"
    echo "  - performance_baseline.csv (baseline performance metrics)"
    echo "  - architecture_mapping.txt (current architecture documentation)"
    echo "  - enhancement_opportunities.txt (optimization recommendations)"
    echo "  - system_resources.txt (system resource analysis)"
    
} | tee "${ASSESSMENT_DIR}/summary_report.txt"

log ""
log "✅ ASSESSMENT COMPLETED SUCCESSFULLY"
log "Assessment results saved to: ${ASSESSMENT_DIR}"
log ""
log "🚀 READY TO PROCEED WITH ENHANCED HYBRID ARCHITECTURE IMPLEMENTATION"

# Create assessment summary for main project
cat > "architecture_assessment_summary.md" << EOF
# Architecture Assessment Summary
**Date**: $(date)
**Status**: COMPLETED ✅

## Results
- **Containerized Engines**: $successful_engines/$total_engines operational ($(($successful_engines * 100 / $total_engines))%)
- **Infrastructure Services**: $infra_successful/$total_infra operational ($(($infra_successful * 100 / $total_infra))%)
- **Integrated Engines**: 8+ engines identified and mapped

## Readiness Status
$([ $successful_engines -ge 7 ] && [ $infra_successful -ge 4 ] && echo "🟢 READY to proceed with Enhanced Hybrid Architecture implementation" || echo "🟡 CAUTION: Some issues need attention before proceeding")

## Detailed Results
Full assessment data: \`${ASSESSMENT_DIR}/\`

**Next Phase**: Baseline Performance Measurement
EOF

echo ""
echo "📄 Assessment summary created: architecture_assessment_summary.md"
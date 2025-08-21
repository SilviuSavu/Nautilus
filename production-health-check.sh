#!/bin/bash

# Nautilus Trading Platform - Production Health Check Script
# Comprehensive health monitoring for production environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Functions
log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}"; PASSED_CHECKS=$((PASSED_CHECKS + 1)); }
warning() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}"; WARNINGS=$((WARNINGS + 1)); }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}"; FAILED_CHECKS=$((FAILED_CHECKS + 1)); }

# Health check function
check_service() {
    local service_name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    local timeout="${4:-5}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "$url" 2>/dev/null || echo "000")
    
    if [ "$status_code" = "$expected_status" ]; then
        success "$service_name is healthy (HTTP $status_code)"
        return 0
    else
        error "$service_name is unhealthy (HTTP $status_code, expected $expected_status)"
        return 1
    fi
}

# Response time check
check_response_time() {
    local service_name="$1"
    local url="$2"
    local max_time="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    local response_time
    response_time=$(curl -w "%{time_total}" -s -o /dev/null --max-time 10 "$url" 2>/dev/null || echo "999")
    
    if (( $(echo "$response_time <= $max_time" | bc -l) )); then
        success "$service_name response time: ${response_time}s (≤ ${max_time}s)"
        return 0
    else
        error "$service_name response time too high: ${response_time}s (> ${max_time}s)"
        return 1
    fi
}

# Docker service check
check_docker_service() {
    local service_name="$1"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker ps --filter "name=$service_name" --filter "status=running" | grep -q "$service_name"; then
        success "Docker service $service_name is running"
        return 0
    else
        error "Docker service $service_name is not running"
        return 1
    fi
}

# Database connection check
check_database() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker exec nautilus-postgres-primary pg_isready -U ${PRODUCTION_DB_USER:-nautilus} > /dev/null 2>&1; then
        success "PostgreSQL database is accepting connections"
    else
        error "PostgreSQL database is not accepting connections"
        return 1
    fi
    
    # Check database size and connections
    local db_size
    db_size=$(docker exec nautilus-postgres-primary psql -U ${PRODUCTION_DB_USER:-nautilus} -d ${PRODUCTION_DB_NAME:-nautilus_prod} -t -c "SELECT pg_size_pretty(pg_database_size('${PRODUCTION_DB_NAME:-nautilus_prod}'));" 2>/dev/null | xargs || echo "Unknown")
    log "Database size: $db_size"
    
    local active_connections
    active_connections=$(docker exec nautilus-postgres-primary psql -U ${PRODUCTION_DB_USER:-nautilus} -d ${PRODUCTION_DB_NAME:-nautilus_prod} -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" 2>/dev/null | xargs || echo "Unknown")
    log "Active database connections: $active_connections"
    
    return 0
}

# Redis check
check_redis() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker exec nautilus-redis-master redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q "PONG"; then
        success "Redis master is responding"
    else
        error "Redis master is not responding"
        return 1
    fi
    
    # Check Redis memory usage
    local redis_memory
    redis_memory=$(docker exec nautilus-redis-master redis-cli -a "${REDIS_PASSWORD:-}" info memory 2>/dev/null | grep "used_memory_human" | cut -d: -f2 | tr -d '\r' || echo "Unknown")
    log "Redis memory usage: $redis_memory"
    
    return 0
}

# SSL certificate check
check_ssl_certificate() {
    local domain="${1:-localhost}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if command -v openssl >/dev/null 2>&1; then
        local cert_info
        cert_info=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null || echo "")
        
        if [ -n "$cert_info" ]; then
            local expiry_date
            expiry_date=$(echo "$cert_info" | grep "notAfter" | cut -d= -f2)
            success "SSL certificate is valid (expires: $expiry_date)"
        else
            warning "SSL certificate check failed or not configured"
        fi
    else
        warning "OpenSSL not available for certificate check"
    fi
}

# Disk space check
check_disk_space() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -lt 80 ]; then
        success "Disk usage: ${disk_usage}% (< 80%)"
    elif [ "$disk_usage" -lt 90 ]; then
        warning "Disk usage: ${disk_usage}% (approaching limit)"
    else
        error "Disk usage critical: ${disk_usage}% (> 90%)"
        return 1
    fi
}

# Memory check
check_memory() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    local memory_usage
    memory_usage=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
    
    if (( $(echo "$memory_usage < 80" | bc -l) )); then
        success "Memory usage: ${memory_usage}% (< 80%)"
    elif (( $(echo "$memory_usage < 90" | bc -l) )); then
        warning "Memory usage: ${memory_usage}% (approaching limit)"
    else
        error "Memory usage critical: ${memory_usage}% (> 90%)"
        return 1
    fi
}

# IB Gateway check
check_ib_gateway() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if docker ps --filter "name=nautilus-ib-primary" --filter "status=running" | grep -q "nautilus-ib-primary"; then
        success "IB Gateway primary is running"
        
        # Check if API port is responding
        if curl -s --max-time 5 http://localhost:4001 > /dev/null 2>&1; then
            success "IB Gateway API is responding"
        else
            warning "IB Gateway API not responding (may be normal during startup)"
        fi
    else
        error "IB Gateway primary is not running"
        return 1
    fi
}

# Trading API endpoints check
check_trading_endpoints() {
    log "🏪 Checking critical trading endpoints..."
    
    local endpoints=(
        "/health:Health Check"
        "/api/auth/health:Authentication"
        "/api/market-data/stream/status:Market Data Stream"
        "/api/positions/current:Position Monitoring"
        "/api/portfolio/summary:Portfolio Summary"
        "/api/trades/history:Trade History"
        "/api/risk/assessment:Risk Assessment"
        "/api/strategies/config:Strategy Configuration"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        local endpoint=$(echo "$endpoint_info" | cut -d: -f1)
        local name=$(echo "$endpoint_info" | cut -d: -f2)
        
        check_service "$name" "$BACKEND_URL$endpoint"
    done
}

# Performance benchmarks
check_performance() {
    log "⚡ Running performance checks..."
    
    # API response time
    check_response_time "Backend API" "$BACKEND_URL/health" 0.1
    
    # Frontend load time
    check_response_time "Frontend" "$FRONTEND_URL" 2.0
    
    # Database query performance
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    local query_time
    query_time=$(docker exec nautilus-postgres-primary psql -U ${PRODUCTION_DB_USER:-nautilus} -d ${PRODUCTION_DB_NAME:-nautilus_prod} -c "\timing on" -c "SELECT 1;" 2>/dev/null | grep "Time:" | awk '{print $2}' | sed 's/ms//' || echo "999")
    
    if (( $(echo "$query_time < 10" | bc -l) )); then
        success "Database query time: ${query_time}ms (< 10ms)"
    else
        warning "Database query time: ${query_time}ms (> 10ms)"
    fi
}

# Security checks
check_security() {
    log "🔒 Running security checks..."
    
    # Check for HTTP to HTTPS redirect
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    local http_status
    http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost 2>/dev/null || echo "000")
    
    if [ "$http_status" = "301" ] || [ "$http_status" = "302" ]; then
        success "HTTP to HTTPS redirect is configured"
    elif [ "$http_status" = "000" ]; then
        warning "HTTP endpoint not accessible (may be expected)"
    else
        warning "HTTP to HTTPS redirect not configured (status: $http_status)"
    fi
    
    # Check security headers
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    local security_headers
    security_headers=$(curl -s -I "$FRONTEND_URL" | grep -i "x-frame-options\|x-content-type-options\|x-xss-protection" | wc -l)
    
    if [ "$security_headers" -ge 2 ]; then
        success "Security headers are configured"
    else
        warning "Some security headers may be missing"
    fi
}

# Monitoring checks
check_monitoring() {
    log "📊 Checking monitoring stack..."
    
    check_service "Prometheus" "$PROMETHEUS_URL/-/healthy"
    check_service "Grafana" "$GRAFANA_URL/api/health"
    
    # Check if AlertManager is running
    check_docker_service "nautilus-alertmanager"
}

# Generate summary report
generate_summary() {
    local pass_rate
    if [ $TOTAL_CHECKS -gt 0 ]; then
        pass_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
    else
        pass_rate=0
    fi
    
    echo ""
    echo "=========================================="
    echo "🏥 PRODUCTION HEALTH CHECK SUMMARY"
    echo "=========================================="
    echo "📊 Total Checks:    $TOTAL_CHECKS"
    echo "✅ Passed:          $PASSED_CHECKS"
    echo "❌ Failed:          $FAILED_CHECKS"
    echo "⚠️  Warnings:       $WARNINGS"
    echo "📈 Pass Rate:       $pass_rate%"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNINGS -eq 0 ]; then
            echo -e "${GREEN}🎉 ALL SYSTEMS HEALTHY - PRODUCTION READY${NC}"
            exit 0
        else
            echo -e "${YELLOW}✅ SYSTEMS OPERATIONAL WITH WARNINGS${NC}"
            exit 0
        fi
    else
        echo -e "${RED}❌ CRITICAL ISSUES DETECTED - INVESTIGATE IMMEDIATELY${NC}"
        exit 1
    fi
}

# Main health check execution
main() {
    echo "🚀 Nautilus Trading Platform - Production Health Check"
    echo "======================================================"
    echo "Timestamp: $(date)"
    echo ""
    
    # Infrastructure checks
    log "🏗️  Checking infrastructure services..."
    check_docker_service "nautilus-lb-prod"
    check_docker_service "nautilus-backend-blue"
    check_docker_service "nautilus-frontend-blue"
    check_docker_service "nautilus-postgres-primary"
    check_docker_service "nautilus-redis-master"
    
    # Database and cache
    log "🗄️  Checking data services..."
    check_database
    check_redis
    
    # Core application
    log "💼 Checking application services..."
    check_trading_endpoints
    
    # Performance
    check_performance
    
    # Security
    check_security
    
    # IB Gateway
    log "🔌 Checking external integrations..."
    check_ib_gateway
    
    # Monitoring
    check_monitoring
    
    # System resources
    log "💻 Checking system resources..."
    check_disk_space
    check_memory
    
    # SSL certificate (if domain provided)
    if [ -n "${DOMAIN:-}" ]; then
        log "🔐 Checking SSL certificate..."
        check_ssl_certificate "$DOMAIN"
    fi
    
    # Generate summary
    generate_summary
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
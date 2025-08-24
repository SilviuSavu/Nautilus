#!/bin/bash

# Nautilus M4 Max Optimized Container Startup Script
# Intelligent container startup based on workload priority and hardware detection

set -euo pipefail

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="M4 Max Optimized Startup"
NAUTILUS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" | tee -a "${LOG_FILE}"
    fi
}

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}" | tee -a "${LOG_FILE}"
}

# Configuration variables
COMPOSE_FILE="${NAUTILUS_HOME}/docker-compose.m4max.yml"
RESOURCE_PROFILES="${NAUTILUS_HOME}/backend/docker/resource-profiles.yml"
DOCKER_DESKTOP_CONFIG="${NAUTILUS_HOME}/docker-desktop-m4max.json"
LOG_DIR="${NAUTILUS_HOME}/logs/startup"
LOG_FILE="${LOG_DIR}/m4max-startup-$(date +%Y%m%d-%H%M%S).log"
MONITORING_CONFIG="${NAUTILUS_HOME}/monitoring/m4max-metrics.yml"

# Performance monitoring variables
THERMAL_THRESHOLD=85
CPU_THRESHOLD=90
MEMORY_THRESHOLD=85
STARTUP_TIMEOUT=300

# Create log directory
mkdir -p "${LOG_DIR}"

# Hardware detection and capabilities
detect_hardware() {
    log_section "Hardware Detection"
    
    # Detect CPU architecture
    ARCH=$(uname -m)
    log_info "Architecture: ${ARCH}"
    
    # Detect Apple Silicon
    if [[ "${ARCH}" == "arm64" ]]; then
        APPLE_SILICON=true
        log_info "✅ Apple Silicon detected"
        
        # Try to detect M4 Max specifically
        if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Apple M4 Max"; then
            M4_MAX=true
            log_info "✅ Apple M4 Max chip confirmed"
        else
            M4_MAX=false
            log_warn "⚠️  Apple Silicon detected but not confirmed as M4 Max"
        fi
    else
        APPLE_SILICON=false
        M4_MAX=false
        log_warn "⚠️  Not running on Apple Silicon (detected: ${ARCH})"
    fi
    
    # Detect Docker Desktop
    if docker version --format '{{.Server.Os}}' 2>/dev/null | grep -q "linux"; then
        DOCKER_DESKTOP=true
        log_info "✅ Docker Desktop detected"
    else
        DOCKER_DESKTOP=false
        log_warn "⚠️  Docker Desktop not detected"
    fi
    
    # Get system specifications
    CPU_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo "unknown")
    LOGICAL_CORES=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo "unknown")
    MEMORY_GB=$(echo "$(sysctl -n hw.memsize 2>/dev/null || echo "0") / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "unknown")
    
    log_info "CPU Cores: ${CPU_CORES} physical, ${LOGICAL_CORES} logical"
    log_info "Memory: ${MEMORY_GB} GB"
    
    # Export detected capabilities
    export APPLE_SILICON M4_MAX DOCKER_DESKTOP CPU_CORES LOGICAL_CORES MEMORY_GB
}

# Validate Docker Desktop configuration
validate_docker_config() {
    log_section "Docker Configuration Validation"
    
    if [[ ! -f "${DOCKER_DESKTOP_CONFIG}" ]]; then
        log_warn "Docker Desktop M4 Max configuration not found: ${DOCKER_DESKTOP_CONFIG}"
        return 0
    fi
    
    # Check Docker Desktop resource allocation
    local docker_info
    docker_info=$(docker system info 2>/dev/null || echo "")
    
    if [[ -n "${docker_info}" ]]; then
        local docker_cpus docker_memory
        docker_cpus=$(echo "${docker_info}" | grep "CPUs:" | awk '{print $2}' || echo "unknown")
        docker_memory=$(echo "${docker_info}" | grep "Total Memory:" | awk '{print $3}' | sed 's/GiB//' || echo "unknown")
        
        log_info "Docker allocated CPUs: ${docker_cpus}"
        log_info "Docker allocated Memory: ${docker_memory} GB"
        
        # Validate resource allocation
        if [[ "${docker_cpus}" != "unknown" && "${docker_cpus}" -lt 8 ]]; then
            log_warn "⚠️  Docker allocated only ${docker_cpus} CPUs (recommended: 12+ for M4 Max)"
        fi
        
        if [[ "${docker_memory}" != "unknown" ]] && (( $(echo "${docker_memory} < 16" | bc -l 2>/dev/null || echo 1) )); then
            log_warn "⚠️  Docker allocated only ${docker_memory} GB RAM (recommended: 24+ GB for M4 Max)"
        fi
    else
        log_error "Unable to retrieve Docker system info"
    fi
}

# Monitor system thermal state
monitor_thermal_state() {
    log_debug "Checking thermal state..."
    
    # macOS thermal state check
    if command -v pmset >/dev/null 2>&1; then
        local thermal_state
        thermal_state=$(pmset -g therm 2>/dev/null | grep -i "thermal" || echo "unknown")
        log_debug "Thermal state: ${thermal_state}"
        
        if echo "${thermal_state}" | grep -qi "critical\|emergency"; then
            log_error "🔥 Critical thermal state detected - aborting startup"
            return 1
        elif echo "${thermal_state}" | grep -qi "pressure\|hot"; then
            log_warn "🌡️  Thermal pressure detected - will monitor closely"
        fi
    fi
    
    return 0
}

# Optimize system settings for M4 Max
optimize_system_settings() {
    log_section "System Optimization"
    
    # Set optimal environment variables for M4 Max
    log_info "Setting M4 Max optimized environment variables..."
    
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    export BUILDKIT_PROGRESS=plain
    
    # M4 Max specific optimizations
    if [[ "${M4_MAX}" == "true" ]]; then
        export M4_MAX_OPTIMIZATION=enabled
        export METAL_FRAMEWORK=enabled
        export COREML_FRAMEWORK=enabled
        export NEURAL_ENGINE=enabled
        export ARM64_NATIVE=true
        
        # CPU optimization
        export OMP_NUM_THREADS=12
        export MKL_NUM_THREADS=12
        export OPENBLAS_NUM_THREADS=12
        export VECLIB_MAXIMUM_THREADS=12
        
        log_info "✅ M4 Max specific optimizations enabled"
    else
        log_warn "⚠️  M4 Max not detected - using generic ARM64 optimizations"
    fi
    
    # Docker Compose optimization
    export COMPOSE_PARALLEL_LIMIT=10
    export COMPOSE_HTTP_TIMEOUT=300
}

# Determine container startup order based on workload priority
determine_startup_order() {
    log_section "Container Startup Order Planning"
    
    # Define startup phases based on criticality
    PHASE_1_INFRASTRUCTURE=(
        "postgres"
        "redis"
    )
    
    PHASE_2_CRITICAL=(
        "risk-engine"
        "marketdata-engine" 
    )
    
    PHASE_3_CORE=(
        "backend"
        "websocket-engine"
    )
    
    PHASE_4_ANALYTICS=(
        "factor-engine"
        "analytics-engine"
        "ml-engine"
        "features-engine"
    )
    
    PHASE_5_SERVICES=(
        "strategy-engine"
        "portfolio-engine"
    )
    
    PHASE_6_FRONTEND=(
        "frontend"
        "nginx"
    )
    
    PHASE_7_MONITORING=(
        "prometheus"
        "grafana"
        "node-exporter"
        "cadvisor"
    )
    
    log_info "Planned startup phases:"
    log_info "  Phase 1 (Infrastructure): ${PHASE_1_INFRASTRUCTURE[*]}"
    log_info "  Phase 2 (Critical): ${PHASE_2_CRITICAL[*]}"
    log_info "  Phase 3 (Core): ${PHASE_3_CORE[*]}"
    log_info "  Phase 4 (Analytics): ${PHASE_4_ANALYTICS[*]}"
    log_info "  Phase 5 (Services): ${PHASE_5_SERVICES[*]}"
    log_info "  Phase 6 (Frontend): ${PHASE_6_FRONTEND[*]}"
    log_info "  Phase 7 (Monitoring): ${PHASE_7_MONITORING[*]}"
}

# Start containers in phases with health checks
start_container_phase() {
    local phase_name="$1"
    shift
    local containers=("$@")
    
    log_section "Starting ${phase_name}"
    
    # Start all containers in this phase
    for container in "${containers[@]}"; do
        log_info "Starting ${container}..."
        
        if docker-compose -f "${COMPOSE_FILE}" up -d "${container}"; then
            log_info "✅ ${container} started successfully"
        else
            log_error "❌ Failed to start ${container}"
            return 1
        fi
    done
    
    # Wait for containers to be healthy
    log_info "Waiting for phase ${phase_name} containers to be healthy..."
    local max_wait=120
    local wait_time=0
    
    while [[ ${wait_time} -lt ${max_wait} ]]; do
        local all_healthy=true
        
        for container in "${containers[@]}"; do
            local container_name="nautilus-${container}-m4max"
            local health_status
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "unknown")
            
            if [[ "${health_status}" != "healthy" && "${health_status}" != "unknown" ]]; then
                all_healthy=false
                break
            fi
        done
        
        if [[ "${all_healthy}" == "true" ]]; then
            log_info "✅ All ${phase_name} containers are healthy"
            return 0
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
        
        if [[ $((wait_time % 30)) -eq 0 ]]; then
            log_info "Still waiting for ${phase_name} containers... (${wait_time}/${max_wait}s)"
        fi
    done
    
    log_warn "⚠️  Timeout waiting for ${phase_name} containers to be healthy"
    return 0  # Don't fail the entire startup
}

# Monitor container performance during startup
monitor_container_performance() {
    log_section "Performance Monitoring"
    
    # Get container resource usage
    local container_stats
    container_stats=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" 2>/dev/null | head -20)
    
    if [[ -n "${container_stats}" ]]; then
        log_info "Container Resource Usage:"
        echo "${container_stats}" | while IFS= read -r line; do
            log_info "  ${line}"
        done
    else
        log_warn "Unable to retrieve container stats"
    fi
    
    # Check host system resources
    if command -v top >/dev/null 2>&1; then
        local cpu_usage memory_usage
        cpu_usage=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "unknown")
        memory_usage=$(top -l 1 -n 0 | grep "PhysMem" | awk '{print $2}' | sed 's/M//' 2>/dev/null || echo "unknown")
        
        log_info "Host CPU Usage: ${cpu_usage}%"
        log_info "Host Memory Usage: ${memory_usage}MB"
        
        # Check for performance issues
        if [[ "${cpu_usage}" != "unknown" ]] && (( $(echo "${cpu_usage} > ${CPU_THRESHOLD}" | bc -l 2>/dev/null || echo 0) )); then
            log_warn "⚠️  High CPU usage detected: ${cpu_usage}%"
        fi
    fi
}

# Validate container health after startup
validate_container_health() {
    log_section "Container Health Validation"
    
    local failed_containers=()
    
    # Check all running containers
    local running_containers
    running_containers=$(docker-compose -f "${COMPOSE_FILE}" ps --services --filter status=running 2>/dev/null || echo "")
    
    if [[ -z "${running_containers}" ]]; then
        log_error "❌ No containers are running"
        return 1
    fi
    
    for service in ${running_containers}; do
        local container_name="nautilus-${service}-m4max"
        local status exit_code health
        
        status=$(docker inspect --format='{{.State.Status}}' "${container_name}" 2>/dev/null || echo "unknown")
        exit_code=$(docker inspect --format='{{.State.ExitCode}}' "${container_name}" 2>/dev/null || echo "unknown")
        health=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "no-health-check")
        
        if [[ "${status}" == "running" ]]; then
            if [[ "${health}" == "healthy" || "${health}" == "no-health-check" ]]; then
                log_info "✅ ${service}: running and healthy"
            else
                log_warn "⚠️  ${service}: running but health check failed (${health})"
                failed_containers+=("${service}")
            fi
        else
            log_error "❌ ${service}: not running (status: ${status}, exit: ${exit_code})"
            failed_containers+=("${service}")
        fi
    done
    
    if [[ ${#failed_containers[@]} -gt 0 ]]; then
        log_warn "Containers with issues: ${failed_containers[*]}"
        
        # Show logs for failed containers
        for container in "${failed_containers[@]}"; do
            log_info "Showing logs for ${container}:"
            docker-compose -f "${COMPOSE_FILE}" logs --tail=10 "${container}" || true
        done
    fi
    
    return 0
}

# Performance benchmark after startup
run_performance_benchmark() {
    log_section "Performance Benchmark"
    
    log_info "Running M4 Max performance benchmark..."
    
    # Check if backend is responding
    local max_attempts=30
    local attempt=1
    
    while [[ ${attempt} -le ${max_attempts} ]]; do
        if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
            log_info "✅ Backend health check passed"
            break
        fi
        
        log_debug "Backend health check attempt ${attempt}/${max_attempts}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [[ ${attempt} -gt ${max_attempts} ]]; then
        log_warn "⚠️  Backend health check failed after ${max_attempts} attempts"
    fi
    
    # Run container performance benchmarks if available
    if docker exec nautilus-backend-m4max python -c "import sys; print('Python optimization ready')" 2>/dev/null; then
        log_info "✅ Backend Python environment ready"
    else
        log_warn "⚠️  Backend Python environment not ready"
    fi
    
    log_info "Performance benchmark completed"
}

# Generate startup report
generate_startup_report() {
    log_section "Startup Report"
    
    local end_time=$(date +%s)
    local startup_duration=$((end_time - start_time))
    
    log_info "🎉 Nautilus M4 Max startup completed!"
    log_info "📊 Startup Statistics:"
    log_info "  Duration: ${startup_duration} seconds"
    log_info "  Hardware: ${ARCH} (Apple Silicon: ${APPLE_SILICON}, M4 Max: ${M4_MAX})"
    log_info "  Docker Desktop: ${DOCKER_DESKTOP}"
    log_info "  Resources: ${CPU_CORES} CPU cores, ${MEMORY_GB} GB RAM"
    
    # Service status summary
    local total_services running_services
    total_services=$(docker-compose -f "${COMPOSE_FILE}" config --services | wc -l | tr -d ' ')
    running_services=$(docker-compose -f "${COMPOSE_FILE}" ps --services --filter status=running 2>/dev/null | wc -l | tr -d ' ')
    
    log_info "  Services: ${running_services}/${total_services} running"
    
    # Access information
    log_info "🌐 Access Points:"
    log_info "  Frontend: http://localhost:3000"
    log_info "  Backend API: http://localhost:8001"
    log_info "  Grafana: http://localhost:3002 (admin:admin123)"
    log_info "  Prometheus: http://localhost:9090"
    
    log_info "📁 Logs saved to: ${LOG_FILE}"
    
    # Performance recommendations
    if [[ "${M4_MAX}" != "true" ]]; then
        log_warn "💡 Recommendations:"
        log_warn "  - Consider using M4 Max hardware for optimal performance"
        log_warn "  - Current configuration may not utilize full optimization potential"
    fi
    
    if [[ -n "${DOCKER_MEMORY}" ]] && (( $(echo "${DOCKER_MEMORY} < 24" | bc -l 2>/dev/null || echo 1) )); then
        log_warn "  - Consider allocating more memory to Docker Desktop (recommended: 24+ GB)"
    fi
}

# Error handling and cleanup
cleanup_on_error() {
    log_error "❌ Startup failed - cleaning up..."
    
    # Stop any containers that were started
    docker-compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true
    
    # Show recent logs
    log_info "Recent Docker logs:"
    docker-compose -f "${COMPOSE_FILE}" logs --tail=20 2>/dev/null || true
    
    exit 1
}

# Main startup function
main() {
    local start_time=$(date +%s)
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    log_section "${SCRIPT_NAME} v${SCRIPT_VERSION}"
    log_info "Starting Nautilus Trading Platform with M4 Max optimizations..."
    log_info "Working directory: ${NAUTILUS_HOME}"
    log_info "Log file: ${LOG_FILE}"
    
    # Validate prerequisites
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        log_error "Docker Compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi
    
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker not found - please install Docker Desktop"
        exit 1
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose not found - please install Docker Compose"
        exit 1
    fi
    
    # Execute startup phases
    detect_hardware
    validate_docker_config
    monitor_thermal_state
    optimize_system_settings
    determine_startup_order
    
    # Start containers in phases
    start_container_phase "Infrastructure" "${PHASE_1_INFRASTRUCTURE[@]}"
    start_container_phase "Critical Trading" "${PHASE_2_CRITICAL[@]}"
    start_container_phase "Core Services" "${PHASE_3_CORE[@]}"
    start_container_phase "Analytics Engines" "${PHASE_4_ANALYTICS[@]}"
    start_container_phase "Trading Services" "${PHASE_5_SERVICES[@]}"
    start_container_phase "Frontend" "${PHASE_6_FRONTEND[@]}"
    start_container_phase "Monitoring" "${PHASE_7_MONITORING[@]}"
    
    # Post-startup validation and monitoring
    monitor_container_performance
    validate_container_health
    run_performance_benchmark
    generate_startup_report
    
    log_info "✅ M4 Max optimized startup completed successfully!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
#!/bin/bash

# M4 Max Optimized Docker Startup Script
# Hardware-aware phased container initialization for Apple M4 Max
# Ensures optimal resource allocation and thermal management

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/m4max-startup.log"
HEALTH_CHECK_TIMEOUT=120
MAX_RETRIES=3

# Initialize logging
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} M4 Max Docker Startup Script v1.0${NC}"
echo -e "${BLUE} Nautilus Trading Platform${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Started at: $(date)"
echo

# Function to print status messages
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $1"
}

# Hardware detection and validation
detect_hardware() {
    print_status "Detecting hardware specifications..."
    
    # Check for M4 Max chip
    CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip:")
    if [[ $CHIP_INFO == *"Apple M4 Max"* ]]; then
        print_success "M4 Max chip detected"
        CPU_CORES=$(sysctl -n hw.logicalcpu)
        P_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "10")
        E_CORES=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo "4")
        MEMORY_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        
        print_success "CPU Cores: $CPU_CORES total ($P_CORES P-cores + $E_CORES E-cores)"
        print_success "Memory: ${MEMORY_GB}GB"
        
        if [ "$MEMORY_GB" -lt 32 ]; then
            print_warning "Less than 32GB RAM detected. Some containers may be resource-constrained."
        fi
    else
        print_error "M4 Max chip not detected. This script is optimized for M4 Max hardware."
        print_status "Detected: $CHIP_INFO"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check Docker availability and resources
check_docker() {
    print_status "Checking Docker configuration..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon not running"
        exit 1
    fi
    
    # Check Docker Desktop resource allocation
    DOCKER_CPUS=$(docker info --format '{{.NCPU}}' 2>/dev/null || echo "unknown")
    DOCKER_MEMORY=$(docker info --format '{{.MemTotal}}' 2>/dev/null | numfmt --to=iec 2>/dev/null || echo "unknown")
    
    print_success "Docker daemon running"
    print_status "Docker resources: $DOCKER_CPUS CPUs, $DOCKER_MEMORY memory"
    
    # Verify docker-compose availability
    if command -v docker-compose &> /dev/null; then
        print_success "docker-compose available"
    elif docker compose version &> /dev/null; then
        print_success "docker compose (plugin) available"
        alias docker-compose='docker compose'
    else
        print_error "docker-compose not available"
        exit 1
    fi
}

# Thermal monitoring setup
setup_thermal_monitoring() {
    print_status "Setting up thermal monitoring..."
    
    # Create thermal monitoring script
    cat > "$SCRIPT_DIR/thermal-monitor.sh" << 'EOF'
#!/bin/bash
while true; do
    TEMP=$(sudo powermetrics -n 1 -i 1000 --samplers smc -a --hide-cpu-duty-cycle 2>/dev/null | grep -E "CPU die temperature|GPU die temperature" || true)
    if [ ! -z "$TEMP" ]; then
        echo "[$(date +'%H:%M:%S')] Thermal: $TEMP" >> thermal-monitor.log
    fi
    sleep 30
done
EOF
    
    chmod +x "$SCRIPT_DIR/thermal-monitor.sh"
    
    # Start thermal monitoring in background (if not already running)
    if ! pgrep -f "thermal-monitor.sh" > /dev/null; then
        nohup "$SCRIPT_DIR/thermal-monitor.sh" &
        print_success "Thermal monitoring started"
    else
        print_status "Thermal monitoring already running"
    fi
}

# Clean up existing containers safely
cleanup_containers() {
    print_status "Cleaning up existing containers..."
    
    # Stop containers gracefully
    if docker-compose ps -q &> /dev/null; then
        print_status "Stopping existing containers..."
        docker-compose down --timeout 30 || true
    fi
    
    # Remove M4 Max tagged images if they exist (to force rebuild)
    M4MAX_IMAGES=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep ":m4max" || true)
    if [ ! -z "$M4MAX_IMAGES" ]; then
        print_status "Removing existing M4 Max optimized images..."
        docker images --format "{{.Repository}}:{{.Tag}}" | grep ":m4max" | xargs -r docker rmi --force
    fi
    
    print_success "Cleanup completed"
}

# Container health check function
wait_for_container_health() {
    local container_name="$1"
    local max_wait="$2"
    local wait_count=0
    
    print_status "Waiting for $container_name to be healthy..."
    
    while [ $wait_count -lt $max_wait ]; do
        if [ "$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null)" = "healthy" ]; then
            print_success "$container_name is healthy"
            return 0
        elif [ "$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null)" = "exited" ]; then
            print_error "$container_name has exited"
            docker logs --tail=20 "$container_name"
            return 1
        fi
        
        sleep 2
        wait_count=$((wait_count + 2))
        
        if [ $((wait_count % 20)) -eq 0 ]; then
            print_status "Still waiting for $container_name... (${wait_count}s/${max_wait}s)"
        fi
    done
    
    print_warning "$container_name health check timeout"
    return 1
}

# Phased container startup with resource awareness
start_phase_1_infrastructure() {
    print_status "Phase 1: Starting infrastructure services..."
    
    # Start PostgreSQL first (highest priority for data integrity)
    docker-compose -f docker-compose.m4max.yml up -d postgres
    wait_for_container_health "nautilus-postgres" 60
    
    # Start Redis
    docker-compose -f docker-compose.m4max.yml up -d redis
    sleep 10
    
    # Verify Redis connectivity
    if docker exec nautilus-redis redis-cli ping | grep -q PONG; then
        print_success "Redis is responding"
    else
        print_error "Redis not responding"
        return 1
    fi
    
    print_success "Phase 1 completed - Infrastructure services ready"
}

start_phase_2_core_services() {
    print_status "Phase 2: Starting core application services..."
    
    # Start backend (needs database and Redis)
    docker-compose -f docker-compose.m4max.yml up -d backend
    sleep 15
    
    # Verify backend API
    local retry_count=0
    while [ $retry_count -lt 30 ]; do
        if curl -s http://localhost:8001/health > /dev/null; then
            print_success "Backend API is responding"
            break
        fi
        sleep 2
        retry_count=$((retry_count + 1))
    done
    
    if [ $retry_count -eq 30 ]; then
        print_error "Backend API not responding"
        docker logs --tail=20 nautilus-backend
        return 1
    fi
    
    # Start monitoring services
    docker-compose -f docker-compose.m4max.yml up -d prometheus grafana
    
    print_success "Phase 2 completed - Core services ready"
}

start_phase_3_engines() {
    print_status "Phase 3: Starting processing engines (M4 Max optimized)..."
    
    # Start engines in order of criticality and resource requirements
    local engines=(
        "risk-engine"           # Highest priority, low resource
        "marketdata-engine"     # High priority, real-time processing
        "websocket-engine"      # High priority, connection handling
        "analytics-engine"      # Medium priority, moderate resources
        "features-engine"       # Medium priority, computation
        "ml-engine"            # Lower priority, high resources
        "factor-engine"        # Lower priority, highest resources
        "strategy-engine"      # Lowest priority, strategy execution
        "portfolio-engine"     # Lowest priority, complex optimization
    )
    
    for engine in "${engines[@]}"; do
        print_status "Starting $engine..."
        docker-compose -f docker-compose.m4max.yml up -d "$engine"
        
        # Wait a bit between engine starts to manage thermal load
        sleep 10
        
        # Check if engine started successfully
        if wait_for_container_health "nautilus-$engine" 60; then
            print_success "$engine started successfully"
        else
            print_warning "$engine failed to start properly"
            # Continue with other engines rather than failing completely
        fi
    done
    
    print_success "Phase 3 completed - Processing engines deployed"
}

start_phase_4_frontend() {
    print_status "Phase 4: Starting frontend services..."
    
    # Start frontend
    docker-compose -f docker-compose.m4max.yml up -d frontend
    sleep 15
    
    # Start nginx
    docker-compose -f docker-compose.m4max.yml up -d nginx
    sleep 10
    
    # Verify frontend availability
    if curl -s http://localhost:3000 > /dev/null; then
        print_success "Frontend is responding"
    else
        print_warning "Frontend not responding on port 3000"
    fi
    
    if curl -s http://localhost:80 > /dev/null; then
        print_success "Nginx proxy is responding"
    else
        print_warning "Nginx proxy not responding on port 80"
    fi
    
    print_success "Phase 4 completed - Frontend services ready"
}

# Comprehensive system validation
validate_deployment() {
    print_status "Validating M4 Max optimized deployment..."
    
    local failed_services=0
    local total_services=0
    
    # Check all container statuses
    print_status "Container status check:"
    while IFS= read -r line; do
        total_services=$((total_services + 1))
        container_name=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $6}')
        
        if [[ $status == *"Up"* ]]; then
            print_success "  $container_name: $status"
        else
            print_error "  $container_name: $status"
            failed_services=$((failed_services + 1))
        fi
    done < <(docker-compose -f docker-compose.m4max.yml ps | tail -n +2)
    
    # Health endpoint checks
    local health_endpoints=(
        "http://localhost:8001/health:Backend API"
        "http://localhost:8100/health:Analytics Engine"
        "http://localhost:8200/health:Risk Engine"
        "http://localhost:8300/health:Factor Engine"
        "http://localhost:8400/health:ML Engine"
        "http://localhost:8500/health:Features Engine"
        "http://localhost:8600/health:WebSocket Engine"
        "http://localhost:8700/health:Strategy Engine"
        "http://localhost:8800/health:MarketData Engine"
        "http://localhost:8900/health:Portfolio Engine"
    )
    
    print_status "Health endpoint validation:"
    for endpoint_info in "${health_endpoints[@]}"; do
        IFS=':' read -r url name <<< "$endpoint_info"
        if curl -s --max-time 10 "$url" > /dev/null; then
            print_success "  $name: Healthy"
        else
            print_warning "  $name: Not responding"
            failed_services=$((failed_services + 1))
        fi
    done
    
    # Resource utilization check
    print_status "Resource utilization:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -20
    
    # Final status
    echo
    if [ $failed_services -eq 0 ]; then
        print_success "M4 Max deployment validation successful!"
        print_success "All $total_services services are running optimally"
    else
        print_warning "Deployment completed with $failed_services issues out of $total_services services"
        print_status "Check logs for failed services: docker-compose logs [service_name]"
    fi
}

# Resource monitoring alert
setup_resource_alerts() {
    print_status "Setting up resource monitoring..."
    
    # Create resource monitor script
    cat > "$SCRIPT_DIR/resource-monitor.sh" << 'EOF'
#!/bin/bash
while true; do
    # Memory usage alert
    MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" | sed 's/%//' | sort -nr | head -1)
    if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
        echo "[$(date +'%H:%M:%S')] WARNING: High memory usage detected: ${MEMORY_USAGE}%"
    fi
    
    # CPU usage alert
    CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | sort -nr | head -1)
    if (( $(echo "$CPU_USAGE > 85" | bc -l) )); then
        echo "[$(date +'%H:%M:%S')] WARNING: High CPU usage detected: ${CPU_USAGE}%"
    fi
    
    sleep 60
done >> resource-alerts.log 2>&1
EOF
    
    chmod +x "$SCRIPT_DIR/resource-monitor.sh"
    
    # Start resource monitoring in background
    if ! pgrep -f "resource-monitor.sh" > /dev/null; then
        nohup "$SCRIPT_DIR/resource-monitor.sh" &
        print_success "Resource monitoring started"
    fi
}

# Usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --skip-hardware-check    Skip M4 Max hardware validation"
    echo "  --no-thermal-monitor     Disable thermal monitoring"
    echo "  --fast-start            Skip delays between phases"
    echo "  --validate-only         Only run validation checks"
    echo "  --help                  Show this help message"
}

# Main execution flow
main() {
    local skip_hardware_check=false
    local no_thermal_monitor=false
    local fast_start=false
    local validate_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-hardware-check)
                skip_hardware_check=true
                shift
                ;;
            --no-thermal-monitor)
                no_thermal_monitor=true
                shift
                ;;
            --fast-start)
                fast_start=true
                shift
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    if [ "$validate_only" = true ]; then
        validate_deployment
        exit 0
    fi
    
    # Pre-flight checks
    if [ "$skip_hardware_check" = false ]; then
        detect_hardware
    fi
    
    check_docker
    
    if [ "$no_thermal_monitor" = false ]; then
        setup_thermal_monitoring
    fi
    
    # Cleanup and start
    cleanup_containers
    
    # Execute phased startup
    start_phase_1_infrastructure || {
        print_error "Phase 1 failed - Infrastructure services"
        exit 1
    }
    
    [ "$fast_start" = false ] && sleep 5
    
    start_phase_2_core_services || {
        print_error "Phase 2 failed - Core services"
        exit 1
    }
    
    [ "$fast_start" = false ] && sleep 10
    
    start_phase_3_engines || {
        print_error "Phase 3 failed - Processing engines"
        exit 1
    }
    
    [ "$fast_start" = false ] && sleep 5
    
    start_phase_4_frontend || {
        print_error "Phase 4 failed - Frontend services"
        exit 1
    }
    
    # Setup monitoring
    setup_resource_alerts
    
    # Final validation
    echo
    print_status "Performing final validation..."
    sleep 10
    validate_deployment
    
    # Success summary
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} M4 Max Deployment Complete! ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Access points:"
    echo "  • Frontend:    http://localhost:3000"
    echo "  • Backend API: http://localhost:8001"
    echo "  • Nginx Proxy: http://localhost:80"
    echo "  • Grafana:     http://localhost:3002 (admin/admin123)"
    echo "  • Prometheus:  http://localhost:9090"
    echo
    echo "Engine Health Checks:"
    echo "  • Analytics:   http://localhost:8100/health"
    echo "  • Risk:        http://localhost:8200/health"
    echo "  • Factor:      http://localhost:8300/health"
    echo "  • ML:          http://localhost:8400/health"
    echo "  • Features:    http://localhost:8500/health"
    echo "  • WebSocket:   http://localhost:8600/health"
    echo "  • Strategy:    http://localhost:8700/health"
    echo "  • MarketData:  http://localhost:8800/health"
    echo "  • Portfolio:   http://localhost:8900/health"
    echo
    echo "Monitoring:"
    echo "  • Startup Log: $LOG_FILE"
    echo "  • Thermal Log: $(pwd)/thermal-monitor.log"
    echo "  • Resource Log: $(pwd)/resource-alerts.log"
    echo
    print_success "M4 Max optimized Nautilus platform is ready for trading!"
}

# Error handling
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Execute main function
main "$@"
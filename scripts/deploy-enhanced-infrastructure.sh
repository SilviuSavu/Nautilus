#!/bin/bash

# Deploy Enhanced Infrastructure for 18 Engines
# Brings up the scaled infrastructure with optimized configurations

set -e

echo "🚀 Deploying Enhanced Infrastructure for 18 Specialized Engines"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if we're in the correct directory
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found. Please run this script from the project root directory."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary data directories..."
    
    mkdir -p data/marketdata_cache
    mkdir -p data/engine_logic  
    mkdir -p data/neural_gpu
    mkdir -p config
    
    # Set proper permissions
    chmod 755 data/marketdata_cache
    chmod 755 data/engine_logic
    chmod 755 data/neural_gpu
    
    log_success "Data directories created"
}

# Deploy infrastructure services
deploy_infrastructure() {
    log_info "Deploying infrastructure services..."
    
    # Stop any existing services
    log_info "Stopping existing services..."
    docker-compose down || true
    
    # Create network first
    log_info "Creating Docker network..."
    docker network create nautilus-network || log_warning "Network may already exist"
    
    # Start PostgreSQL first
    log_info "Starting PostgreSQL with enhanced configuration..."
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Test PostgreSQL connection
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD=nautilus123 psql -h localhost -U nautilus -d nautilus -c "SELECT 1;" > /dev/null 2>&1; then
            log_success "PostgreSQL is ready"
            break
        else
            log_info "Waiting for PostgreSQL... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "PostgreSQL failed to start within expected time"
        exit 1
    fi
    
    # Start Redis services
    log_info "Starting Redis message bus architecture..."
    docker-compose up -d redis marketdata-bus engine-logic-bus neural-gpu-bus
    
    # Wait for Redis services
    log_info "Waiting for Redis services to be ready..."
    sleep 5
    
    # Test Redis connections
    redis_ports=(6379 6380 6381 6382)
    for port in "${redis_ports[@]}"; do
        max_attempts=15
        attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            if redis-cli -h localhost -p "$port" ping > /dev/null 2>&1; then
                log_success "Redis on port $port is ready"
                break
            else
                log_info "Waiting for Redis on port $port... (attempt $attempt/$max_attempts)"
                sleep 1
                ((attempt++))
            fi
        done
        
        if [ $attempt -gt $max_attempts ]; then
            log_error "Redis on port $port failed to start"
            exit 1
        fi
    done
    
    # Start monitoring services
    log_info "Starting monitoring services..."
    docker-compose up -d prometheus grafana cadvisor node-exporter redis-exporter postgres-exporter
    
    log_success "Infrastructure services deployed successfully"
}

# Validate deployment
validate_deployment() {
    log_info "Validating infrastructure deployment..."
    
    # Run the infrastructure scaling test
    if [ -f "scripts/test-infrastructure-scaling.sh" ]; then
        log_info "Running infrastructure validation tests..."
        ./scripts/test-infrastructure-scaling.sh
    else
        log_warning "Infrastructure test script not found, running basic validation..."
        
        # Basic validation
        # Check PostgreSQL
        if PGPASSWORD=nautilus123 psql -h localhost -U nautilus -d nautilus -c "SELECT version();" > /dev/null 2>&1; then
            log_success "PostgreSQL is operational"
        else
            log_error "PostgreSQL validation failed"
            return 1
        fi
        
        # Check Redis buses
        for port in 6379 6380 6381 6382; do
            if redis-cli -h localhost -p "$port" ping > /dev/null 2>&1; then
                log_success "Redis on port $port is operational"
            else
                log_error "Redis on port $port validation failed"
                return 1
            fi
        done
    fi
    
    log_success "Infrastructure validation completed"
}

# Display deployment summary
display_summary() {
    echo
    log_success "🎉 Enhanced Infrastructure Deployment Complete!"
    echo
    echo "📊 Infrastructure Summary:"
    echo "=================================="
    echo
    echo "🗄️  PostgreSQL Database:"
    echo "   - Port: 5432"
    echo "   - Max Connections: 500 (scaled for 18 engines)"
    echo "   - Memory: 16GB allocated"
    echo "   - Connection: postgresql://nautilus:nautilus123@localhost:5432/nautilus"
    echo
    echo "🚌 Redis Message Bus Architecture:"
    echo "   - Primary Redis (6379): General operations"
    echo "   - MarketData Bus (6380): Neural Engine optimized data distribution"
    echo "   - Engine Logic Bus (6381): Metal GPU optimized inter-engine communication"
    echo "   - Neural-GPU Bus (6382): Quantum/Physics engine coordination"
    echo
    echo "📊 Monitoring Services:"
    echo "   - Prometheus: http://localhost:9090"
    echo "   - Grafana: http://localhost:3002 (admin/admin123)"
    echo "   - cAdvisor: http://localhost:8080"
    echo
    echo "🔧 Configuration Files:"
    echo "   - Data Lake Balancing: config/data-lake-balancing-strategy.yml"
    echo "   - Connection Optimization: config/connection-persistence-optimization.yml"
    echo "   - Neural-GPU Redis Config: backend/config/redis-neural-gpu-optimized.conf"
    echo
    echo "🎯 Engine Deployment Ready:"
    echo "   - Infrastructure supports 18 specialized engines"
    echo "   - Connection pooling: 54-144 database connections"
    echo "   - Triple message bus architecture operational"
    echo "   - M4 Max hardware acceleration ready"
    echo
    echo "🚀 Next Steps:"
    echo "   1. Deploy engines using their specific startup scripts"
    echo "   2. Monitor performance via Grafana dashboards"
    echo "   3. Verify engine health endpoints"
    echo "   4. Configure data source connections"
    echo
    echo "✅ Infrastructure is ready for 18 specialized engines!"
}

# Error handling
handle_error() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose down || true
    exit 1
}

# Set error trap
trap handle_error ERR

# Main deployment function
main() {
    log_info "Starting enhanced infrastructure deployment..."
    
    check_prerequisites
    create_directories
    deploy_infrastructure
    validate_deployment
    display_summary
    
    log_success "Enhanced infrastructure deployment completed successfully!"
}

# Run main function
main "$@"
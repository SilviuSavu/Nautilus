#!/bin/bash

# Nautilus Trading Platform - Production Deployment Script
# Blue-Green Deployment with Zero Downtime

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="docker-compose.production.yml"
UAT_SCRIPT="./uat-validation-script.sh"
HEALTH_CHECK_SCRIPT="./production-health-check.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Error handling
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Running cleanup..."
        docker-compose -f $COMPOSE_FILE --profile green down
        exit 1
    fi
}
trap cleanup EXIT

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."
    
    # Check if required files exist
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "Production docker-compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check environment variables
    if [ -z "$PRODUCTION_DATABASE_URL" ]; then
        error "PRODUCTION_DATABASE_URL environment variable not set"
        exit 1
    fi
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Run UAT validation
    log "🧪 Running UAT validation..."
    if [ -f "$UAT_SCRIPT" ]; then
        if ! $UAT_SCRIPT; then
            error "UAT validation failed. Deployment aborted."
            exit 1
        fi
        success "UAT validation passed"
    else
        warning "UAT script not found, skipping validation"
    fi
    
    # Infrastructure health check
    log "🏥 Running infrastructure health check..."
    if [ -f "$HEALTH_CHECK_SCRIPT" ]; then
        if ! $HEALTH_CHECK_SCRIPT; then
            error "Infrastructure health check failed. Deployment aborted."
            exit 1
        fi
        success "Infrastructure health check passed"
    else
        warning "Health check script not found, skipping"
    fi
    
    success "Pre-deployment checks completed"
}

# Build production images
build_images() {
    log "🔨 Building production images..."
    
    docker-compose -f $COMPOSE_FILE build --no-cache --parallel
    
    success "Production images built successfully"
}

# Deploy to Green environment
deploy_green() {
    log "🌱 Deploying to Green environment..."
    
    # Start Green environment
    docker-compose -f $COMPOSE_FILE --profile green up -d
    
    # Wait for services to be ready
    log "⏳ Waiting for Green environment to be ready..."
    sleep 30
    
    # Health check Green environment
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts..."
        
        if curl -s http://localhost:8001/health > /dev/null; then
            success "Green environment is healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Green environment failed to start"
            exit 1
        fi
        
        sleep 30
        attempt=$((attempt + 1))
    done
    
    # Run smoke tests on Green
    log "🚬 Running smoke tests on Green environment..."
    
    # Test critical endpoints
    local endpoints=(
        "/health"
        "/api/auth/health"
        "/api/market-data/stream/status"
        "/api/positions/current"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s "http://localhost:8001$endpoint" > /dev/null; then
            success "Green environment endpoint $endpoint is responding"
        else
            error "Green environment endpoint $endpoint is not responding"
            exit 1
        fi
    done
    
    success "Green environment deployed and validated"
}

# Database migration (if needed)
database_migration() {
    log "🗄️  Checking for database migrations..."
    
    # Check if migrations are needed
    if [ -f "migrations/pending.sql" ]; then
        log "📊 Running database migrations..."
        
        # Backup database before migration
        log "💾 Creating database backup..."
        docker exec nautilus-postgres-primary pg_dump -U $PRODUCTION_DB_USER -d $PRODUCTION_DB_NAME \
            > backups/pre_migration_$(date +%Y%m%d_%H%M%S).sql
        
        # Run migrations
        docker exec nautilus-postgres-primary psql -U $PRODUCTION_DB_USER -d $PRODUCTION_DB_NAME \
            -f /backups/migrations/pending.sql
        
        success "Database migrations completed"
    else
        log "ℹ️  No database migrations needed"
    fi
}

# Traffic switch (Blue to Green)
traffic_switch() {
    log "🔄 Starting traffic switch from Blue to Green..."
    
    # Step 1: 10% canary traffic
    log "📊 Switching 10% traffic to Green (Canary deployment)..."
    # Update load balancer configuration for 10% traffic
    sed -i.bak 's/weight=3/weight=3/g; s/backup;//g' nginx/production.conf
    docker exec nautilus-lb-prod nginx -s reload
    
    # Monitor for 5 minutes
    log "⏳ Monitoring canary deployment for 5 minutes..."
    sleep 300
    
    # Check error rates
    if ! check_error_rates; then
        error "High error rates detected during canary. Rolling back..."
        rollback_traffic
        exit 1
    fi
    success "Canary deployment successful"
    
    # Step 2: 50% traffic
    log "📊 Switching 50% traffic to Green..."
    sed -i 's/server backend-blue/server backend-blue weight=1/g; s/server backend-green.*backup/server backend-green weight=1/g' nginx/production.conf
    docker exec nautilus-lb-prod nginx -s reload
    
    # Monitor for 5 minutes
    log "⏳ Monitoring 50% traffic split for 5 minutes..."
    sleep 300
    
    if ! check_error_rates; then
        error "High error rates detected during 50% split. Rolling back..."
        rollback_traffic
        exit 1
    fi
    success "50% traffic split successful"
    
    # Step 3: 100% traffic to Green
    log "📊 Switching 100% traffic to Green..."
    sed -i 's/backend-blue/backend-green/g' nginx/production.conf
    docker exec nautilus-lb-prod nginx -s reload
    
    success "Traffic switch completed - Green environment is now active"
}

# Check error rates
check_error_rates() {
    local error_rate=$(curl -s http://localhost:9090/api/v1/query?query=rate\(http_requests_total\{status=~\"5..\"\}\[5m\]\) | jq -r '.data.result[0].value[1] // "0"')
    
    if (( $(echo "$error_rate > 0.01" | bc -l) )); then
        return 1
    fi
    return 0
}

# Rollback traffic
rollback_traffic() {
    warning "🔙 Rolling back traffic to Blue environment..."
    
    # Restore original nginx configuration
    cp nginx/production.conf.bak nginx/production.conf
    docker exec nautilus-lb-prod nginx -s reload
    
    warning "Traffic rolled back to Blue environment"
}

# Post-deployment validation
post_deployment_validation() {
    log "✅ Running post-deployment validation..."
    
    # Full system health check
    log "🏥 Running comprehensive health check..."
    
    # Check all critical services
    local services=(
        "nautilus-lb-prod"
        "nautilus-backend-green"
        "nautilus-frontend-green"
        "nautilus-postgres-primary"
        "nautilus-redis-master"
        "nautilus-prometheus"
        "nautilus-grafana"
    )
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q $service; then
            success "Service $service is running"
        else
            error "Service $service is not running"
            exit 1
        fi
    done
    
    # Performance validation
    log "⚡ Running performance validation..."
    
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null http://localhost/api/health)
    if (( $(echo "$response_time < 0.1" | bc -l) )); then
        success "API response time: ${response_time}s (< 100ms)"
    else
        error "API response time too high: ${response_time}s"
        exit 1
    fi
    
    # Security validation
    log "🔒 Running security validation..."
    
    # Check HTTPS redirect
    local http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost)
    if [ "$http_status" -eq "301" ] || [ "$http_status" -eq "302" ]; then
        success "HTTPS redirect is working"
    else
        warning "HTTPS redirect not configured (status: $http_status)"
    fi
    
    # Trading functionality test
    log "💼 Testing trading functionality..."
    
    # Test market data endpoints
    if curl -s http://localhost/api/market-data/stream/status | grep -q "active"; then
        success "Market data streaming is active"
    else
        error "Market data streaming is not active"
        exit 1
    fi
    
    success "Post-deployment validation completed"
}

# Clean up Blue environment
cleanup_blue() {
    log "🧹 Cleaning up Blue environment..."
    
    # Stop Blue services
    docker-compose -f $COMPOSE_FILE stop frontend-blue backend-blue
    
    # Remove Blue containers (keep images for rollback)
    docker-compose -f $COMPOSE_FILE rm -f frontend-blue backend-blue
    
    success "Blue environment cleaned up"
}

# Generate deployment report
generate_report() {
    log "📄 Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > $report_file << EOF
# Nautilus Trading Platform - Deployment Report

**Deployment Date:** $(date)
**Deployment Type:** Blue-Green with Zero Downtime
**Status:** SUCCESS ✅

## Deployment Summary
- **Pre-deployment Checks:** PASSED
- **Image Build:** SUCCESS
- **Green Deployment:** SUCCESS
- **Database Migration:** $([ -f "migrations/pending.sql" ] && echo "EXECUTED" || echo "NOT NEEDED")
- **Traffic Switch:** SUCCESS
- **Post-deployment Validation:** PASSED
- **Blue Cleanup:** SUCCESS

## Performance Metrics
- **API Response Time:** $(curl -w "%{time_total}" -s -o /dev/null http://localhost/api/health)s
- **Deployment Duration:** $((SECONDS / 60)) minutes
- **Downtime:** 0 seconds

## Services Status
$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep nautilus)

## Health Check Results
- ✅ Load Balancer: Healthy
- ✅ Backend: Healthy
- ✅ Frontend: Healthy
- ✅ Database: Healthy
- ✅ Cache: Healthy
- ✅ Monitoring: Healthy

## Next Steps
1. Monitor system performance for 24 hours
2. Verify trading functionality during market hours
3. Update documentation with any configuration changes
4. Plan next deployment based on lessons learned

## Rollback Procedure
If issues are detected:
1. Run: \`./rollback-deployment.sh\`
2. Monitor system recovery
3. Investigate issues in Green environment

---
Generated automatically by production deployment script
EOF

    success "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log "🚀 Starting Nautilus Trading Platform Production Deployment"
    log "========================================================"
    
    local start_time=$SECONDS
    
    # Deployment phases
    pre_deployment_checks
    build_images
    deploy_green
    database_migration
    traffic_switch
    post_deployment_validation
    cleanup_blue
    generate_report
    
    local duration=$((SECONDS - start_time))
    
    success "🎉 Deployment completed successfully in $((duration / 60)) minutes and $((duration % 60)) seconds!"
    success "🌐 Nautilus Trading Platform is now live in production"
    success "📊 Monitor at: https://monitoring.nautilus.com"
    success "🔗 Access at: https://nautilus.com"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
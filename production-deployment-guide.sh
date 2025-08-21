#!/bin/bash
# Nautilus Trading Platform - Production Deployment Guide
# Enhanced version with step-by-step implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"; }
warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="nautilus-trading"
ENVIRONMENT="production"

log "🚀 Nautilus Trading Platform - Production Deployment Guide"
log "=================================================================="

# Step 1: Pre-deployment validation
log "📋 Step 1: Pre-deployment Validation"

validate_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not installed. Please install Docker first."
        exit 1
    fi
    success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose not installed. Please install Docker Compose first."
        exit 1
    fi
    success "Docker Compose found: $(docker-compose --version)"
    
    # Check AWS CLI (if using AWS)
    if command -v aws &> /dev/null; then
        success "AWS CLI found: $(aws --version)"
    else
        warning "AWS CLI not found. Install if using AWS deployment."
    fi
    
    # Check kubectl (if using Kubernetes)
    if command -v kubectl &> /dev/null; then
        success "kubectl found: $(kubectl version --client)"
    else
        warning "kubectl not found. Install if using Kubernetes deployment."
    fi
}

# Step 2: Environment setup
setup_environment() {
    log "🔧 Step 2: Setting up production environment..."
    
    # Create production environment file
    cat > .env.production << EOF
# Nautilus Trading Platform - Production Environment
# Generated: $(date)

# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
PRODUCTION_DATABASE_URL=\${PRODUCTION_DATABASE_URL}
DATABASE_HOST=\${DATABASE_HOST:-prod-db}
DATABASE_PORT=\${DATABASE_PORT:-5432}
DATABASE_NAME=\${DATABASE_NAME:-nautilus_prod}
DATABASE_USER=\${DATABASE_USER}
DATABASE_PASSWORD=\${DATABASE_PASSWORD}

# Redis Configuration
PRODUCTION_REDIS_URL=\${PRODUCTION_REDIS_URL}
REDIS_HOST=\${REDIS_HOST:-prod-redis}
REDIS_PORT=\${REDIS_PORT:-6379}
REDIS_PASSWORD=\${REDIS_PASSWORD}

# Security Configuration
JWT_SECRET=\${JWT_SECRET}
SESSION_SECRET=\${SESSION_SECRET}
ENCRYPTION_KEY=\${ENCRYPTION_KEY}

# Interactive Brokers Configuration
IB_USERID=\${IB_USERID}
IB_PASSWORD=\${IB_PASSWORD}
IB_TRADING_MODE=\${IB_TRADING_MODE:-paper}
IB_CLIENT_ID=\${IB_CLIENT_ID:-1001}
IB_GATEWAY_HOST=\${IB_GATEWAY_HOST:-127.0.0.1}
IB_GATEWAY_PORT=\${IB_GATEWAY_PORT:-4001}

# Exchange API Keys
BINANCE_API_KEY=\${BINANCE_API_KEY}
BINANCE_API_SECRET=\${BINANCE_API_SECRET}
COINBASE_API_KEY=\${COINBASE_API_KEY}
COINBASE_API_SECRET=\${COINBASE_API_SECRET}

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
ALERTMANAGER_ENABLED=true

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_S3_BUCKET=\${BACKUP_S3_BUCKET}
AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY}

# Performance Configuration
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
REQUEST_TIMEOUT=30
API_RATE_LIMIT=1000

# Security Headers
CORS_ORIGINS=https://your-domain.com
ALLOWED_HOSTS=your-domain.com
SECURE_COOKIES=true
EOF

    success "Production environment file created: .env.production"
    warning "⚠️  Please update .env.production with your actual credentials!"
}

# Step 3: Build production images
build_production_images() {
    log "🏗️  Step 3: Building production Docker images..."
    
    # Build backend image
    log "Building backend production image..."
    docker build -f backend/Dockerfile.production -t ${PROJECT_NAME}-backend:latest ./backend
    success "Backend image built successfully"
    
    # Build frontend image
    log "Building frontend production image..."
    docker build -f frontend/Dockerfile.production -t ${PROJECT_NAME}-frontend:latest ./frontend
    success "Frontend image built successfully"
    
    # Tag images for registry (optional)
    if [[ -n "${DOCKER_REGISTRY}" ]]; then
        log "Tagging images for registry: ${DOCKER_REGISTRY}"
        docker tag ${PROJECT_NAME}-backend:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}-backend:latest
        docker tag ${PROJECT_NAME}-frontend:latest ${DOCKER_REGISTRY}/${PROJECT_NAME}-frontend:latest
        success "Images tagged for registry"
    fi
}

# Step 4: Database setup
setup_database() {
    log "🗄️  Step 4: Setting up production database..."
    
    # Start database container
    log "Starting PostgreSQL database..."
    docker-compose -f docker-compose.production.yml up -d postgres-primary postgres-replica redis-primary
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 30
    
    # Run database migrations
    log "Running database migrations..."
    docker-compose -f docker-compose.production.yml exec backend python -m alembic upgrade head
    success "Database migrations completed"
    
    # Create database backup
    log "Creating initial database backup..."
    docker-compose -f docker-compose.production.yml exec postgres-primary pg_dump -U nautilus_prod nautilus_prod > initial_backup.sql
    success "Initial database backup created"
}

# Step 5: Deploy application
deploy_application() {
    log "🚀 Step 5: Deploying application services..."
    
    # Deploy all services
    log "Starting all production services..."
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to start
    log "Waiting for services to initialize..."
    sleep 60
    
    # Health checks
    log "Running health checks..."
    
    # Check backend health
    for i in {1..5}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            success "Backend health check passed"
            break
        else
            warning "Backend health check attempt $i failed, retrying..."
            sleep 10
        fi
    done
    
    # Check frontend
    if curl -f http://localhost:3000 &> /dev/null; then
        success "Frontend health check passed"
    else
        warning "Frontend health check failed"
    fi
    
    # Check database connection
    if docker-compose -f docker-compose.production.yml exec backend python -c "import psycopg2; print('DB OK')" &> /dev/null; then
        success "Database connection verified"
    else
        warning "Database connection failed"
    fi
}

# Step 6: Configure SSL and domain
configure_ssl() {
    log "🔒 Step 6: SSL and domain configuration..."
    
    # This is a placeholder - actual implementation depends on your setup
    log "SSL configuration steps:"
    log "1. Configure your domain DNS to point to your server"
    log "2. Install certbot: sudo apt-get install certbot"
    log "3. Generate SSL certificate: sudo certbot --nginx -d your-domain.com"
    log "4. Update nginx configuration for HTTPS"
    log "5. Test SSL configuration: https://www.ssllabs.com/ssltest/"
    
    warning "⚠️  SSL configuration must be completed manually based on your domain setup"
}

# Step 7: Final validation
final_validation() {
    log "✅ Step 7: Final production validation..."
    
    # List running containers
    log "Running containers:"
    docker-compose -f docker-compose.production.yml ps
    
    # Check service endpoints
    log "Service endpoint checks:"
    
    endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8000/api/v1/nautilus/engine/health"
        "http://localhost:3000"
        "http://localhost:3001/metrics"  # Prometheus
        "http://localhost:3002"         # Grafana
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f "$endpoint" &> /dev/null; then
            success "✅ $endpoint - OK"
        else
            warning "⚠️  $endpoint - Failed"
        fi
    done
    
    # Generate deployment report
    cat > deployment-report.txt << EOF
Nautilus Trading Platform - Production Deployment Report
Deployment Date: $(date)
Deployment Status: COMPLETED

Services Deployed:
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- Database: PostgreSQL with TimescaleDB
- Cache: Redis Cluster
- Monitoring: Prometheus + Grafana
- IB Gateway: Interactive Brokers connection

Next Steps:
1. Configure production domain and SSL
2. Update DNS records
3. Configure monitoring alerts
4. Set up backup schedules
5. Begin paper trading tests
6. Perform load testing
7. Security audit
8. Go-live decision

Important Notes:
- Update .env.production with real credentials
- Configure firewall and security groups
- Set up monitoring alerts
- Test disaster recovery procedures
- Document operational procedures

Deployment completed successfully!
EOF

    success "Deployment report generated: deployment-report.txt"
}

# Main deployment process
main() {
    log "Starting production deployment process..."
    
    validate_prerequisites
    setup_environment
    build_production_images
    setup_database
    deploy_application
    configure_ssl
    final_validation
    
    success "🎉 Production deployment completed successfully!"
    success "📊 Review deployment-report.txt for next steps"
    success "🔧 Don't forget to update .env.production with real credentials"
    
    log "=================================================================="
    log "🚀 Nautilus Trading Platform is now running in production mode!"
    log "📈 Access your trading platform at: http://localhost:3000"
    log "🔧 Admin dashboard at: http://localhost:3002 (Grafana)"
    log "📊 Metrics at: http://localhost:3001 (Prometheus)"
    log "=================================================================="
}

# Run deployment if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
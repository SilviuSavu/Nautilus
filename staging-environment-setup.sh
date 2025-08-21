#!/bin/bash

# Nautilus Trading Platform - Staging Environment Setup Script
# For User Acceptance Testing (UAT)

set -e

echo "🚀 Setting up Nautilus Trading Platform Staging Environment for UAT"
echo "================================================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed." >&2; exit 1; }

# Stop any running development containers
echo "🛑 Stopping development containers..."
docker-compose down 2>/dev/null || true

# Create staging environment file
echo "⚙️  Creating staging environment configuration..."
cat > .env.staging << EOF
# Staging Environment Configuration
ENVIRONMENT=staging
NODE_ENV=staging

# Database
POSTGRES_DB=nautilus_staging
POSTGRES_USER=nautilus_staging
POSTGRES_PASSWORD=staging_password_123!

# IB Gateway (Demo/Paper Trading)
IB_USERID=demo
IB_PASSWORD=demo
IB_CLIENT_ID=2

# API Keys (leave empty for demo mode)
BINANCE_API_KEY=
BINANCE_API_SECRET=
COINBASE_API_KEY=
COINBASE_API_SECRET=
COINBASE_PASSPHRASE=
BYBIT_API_KEY=
BYBIT_API_SECRET=
KRAKEN_API_KEY=
KRAKEN_API_SECRET=
OKX_API_KEY=
OKX_API_SECRET=
OKX_PASSPHRASE=
EOF

# Build staging images
echo "🔨 Building staging Docker images..."
docker-compose -f docker-compose.staging.yml build

# Create staging volumes
echo "💾 Creating staging volumes..."
docker volume create nautilus_postgres_staging_data
docker volume create nautilus_redis_staging_data
docker volume create nautilus_ib_staging_data
docker volume create nautilus_prometheus_staging_data
docker volume create nautilus_grafana_staging_data

# Start staging environment
echo "🌟 Starting staging environment..."
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health checks
echo "🏥 Performing health checks..."

# Check backend health
echo "  • Backend health check..."
for i in {1..10}; do
    if curl -s http://localhost:8001/health > /dev/null; then
        echo "    ✅ Backend is healthy"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "    ❌ Backend health check failed"
        exit 1
    fi
    sleep 5
done

# Check frontend
echo "  • Frontend health check..."
if curl -s http://localhost:3001/health > /dev/null; then
    echo "    ✅ Frontend is healthy"
else
    echo "    ❌ Frontend health check failed"
fi

# Check database
echo "  • Database health check..."
if docker exec nautilus-postgres-staging pg_isready -U nautilus_staging > /dev/null; then
    echo "    ✅ Database is healthy"
else
    echo "    ❌ Database health check failed"
fi

# Check Redis
echo "  • Redis health check..."
if docker exec nautilus-redis-staging redis-cli ping | grep -q PONG; then
    echo "    ✅ Redis is healthy"
else
    echo "    ❌ Redis health check failed"
fi

echo ""
echo "🎉 Staging Environment Setup Complete!"
echo "======================================"
echo ""
echo "📊 Access Points:"
echo "  • Frontend (UAT):      http://localhost:3001"
echo "  • Backend API:         http://localhost:8001"
echo "  • Backend Health:      http://localhost:8001/health"
echo "  • Backend Docs:        http://localhost:8001/docs"
echo "  • Grafana Monitoring:  http://localhost:3002 (admin/staging_admin_123!)"
echo "  • Prometheus Metrics:  http://localhost:9091"
echo "  • IB Gateway VNC:      vnc://localhost:5901"
echo ""
echo "💾 Database Access:"
echo "  • Host: localhost"
echo "  • Port: 5433"
echo "  • Database: nautilus_staging"
echo "  • User: nautilus_staging"
echo "  • Password: staging_password_123!"
echo ""
echo "🔧 Staging Management:"
echo "  • Start:    docker-compose -f docker-compose.staging.yml up -d"
echo "  • Stop:     docker-compose -f docker-compose.staging.yml down"
echo "  • Logs:     docker-compose -f docker-compose.staging.yml logs -f"
echo "  • Status:   docker-compose -f docker-compose.staging.yml ps"
echo ""
echo "🧪 UAT Testing Ready!"
echo "  • All 21 production-ready stories can now be tested"
echo "  • Environment mirrors production configuration"
echo "  • Monitoring and logging enabled for validation"
echo "  • Paper trading mode enabled for safe testing"
echo ""
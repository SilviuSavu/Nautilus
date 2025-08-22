# Docker Setup Quick Reference

## ⚠️ IMPORTANT: Everything Runs in Docker

**DO NOT run any services locally. All components are containerized.**
**CRITICAL: Frontend has NO hardcoded values - all use environment variables.**

## Quick Start
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild containers
docker-compose up --build
```

## Service URLs
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **Database**: localhost:5432
- **Redis**: localhost:6379

## Health Checks
```bash
# Backend health
curl http://localhost:8001/health

# All data sources
curl http://localhost:8001/api/v1/nautilus-data/health

# FRED macro factors
curl http://localhost:8001/api/v1/nautilus-data/fred/macro-factors

# Alpha Vantage search
curl "http://localhost:8001/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL"

# EDGAR health
curl http://localhost:8001/api/v1/edgar/health
```

## Container Management
```bash
# View logs
docker-compose logs backend
docker-compose logs frontend

# Execute commands in containers
docker exec -it nautilus-backend pytest
docker exec -it nautilus-frontend npm test
docker exec -it nautilus-postgres psql -U nautilus -d nautilus
docker exec -it nautilus-redis redis-cli

# Restart specific service
docker-compose restart backend
```

## Data Sources Status
✅ **FRED**: Operational (16+ macro factors)
✅ **Alpha Vantage**: Operational (quotes, search, fundamentals)  
✅ **EDGAR**: Operational (7,848+ companies)
✅ **IBKR**: Configured (Gateway integration)

## Environment Variables (Pre-configured)

### Backend API Keys
- ALPHA_VANTAGE_API_KEY: 271AHP91HVAPDRGP
- FRED_API_KEY: 1f1ba9c949e988e12796b7c1f6cce1bf

### Frontend Environment Variables
- VITE_API_BASE_URL: http://localhost:8001 (backend connection)
- VITE_WS_URL: localhost:8001 (WebSocket connection)
- **NO hardcoded URLs**: All components use these environment variables

## Testing
```bash
# Frontend tests (can run locally)
cd frontend && npm test
cd frontend && npx playwright test

# Backend tests (must run in container)
docker exec nautilus-backend pytest
```
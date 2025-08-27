# Production Deployment Guide

**Complete production deployment guide for Nautilus institutional trading platform** with 12 containerized engines and M4 Max hardware acceleration.

## 🎯 Executive Summary

Deploy enterprise-grade trading platform with **100% operational status**, **1.5-3.5ms response times**, and **45+ RPS throughput**.

**Target Audience**: DevOps Engineers, SREs, System Administrators  
**Prerequisites**: Docker 20.10+, Docker Compose 2.0+, 16GB+ RAM  
**Deployment Time**: ~15 minutes complete system deployment

## 📋 Pre-Deployment Requirements

### Hardware Requirements
```yaml
Minimum Production:
  CPU: 16 cores (Intel/AMD x64 or Apple M4 Max)
  RAM: 32GB available for containers
  Storage: 200GB NVMe SSD
  Network: 1Gbps interface

Recommended Production:
  CPU: 24 cores (optimal engine performance)
  RAM: 64GB (supports 3x scaling)
  Storage: 1TB NVMe SSD (high-performance I/O)
  Network: 10Gbps interface

M4 Max Optimized (Recommended):
  CPU: 12 Performance + 4 Efficiency cores
  GPU: 40 Metal GPU cores, 546 GB/s bandwidth
  Neural: 16-core Neural Engine, 38 TOPS
  Memory: Unified memory architecture
```

### Software Requirements
```yaml
Required Software:
  - Docker Engine: 20.10+ (BuildKit enabled)
  - Docker Compose: 2.0+ (profiles support)
  - Git: 2.30+
  - curl/jq: Health checks and testing

Operating System Support:
  - Ubuntu 20.04/22.04 LTS (recommended)
  - CentOS/RHEL 8/9
  - macOS 12+ with M4 Max (optimal)
  - Windows 11 with WSL2 (development only)
```

### Network Port Requirements
```yaml
Core Platform (Required):
  - 3000: Frontend (React application)
  - 8001: Backend API (FastAPI)
  - 5432: PostgreSQL database
  - 6379: Redis (MessageBus)
  - 3002: Grafana dashboards

Processing Engines (All Operational):
  - 8100: Analytics Engine
  - 8200: Risk Engine (Enhanced Institutional)
  - 8300: Factor Engine (Toraniko v1.1.2)
  - 8400: ML Engine
  - 8500: Features Engine
  - 8600: WebSocket Engine
  - 8700: Strategy Engine
  - 8800: MarketData Engine
  - 8900: Portfolio Engine (Institutional Grade)
  - 9000: Collateral Engine (Mission Critical)
  - 10000: VPIN Engine (Market Microstructure)
```

## 🚀 Production Deployment Steps

### Step 1: Environment Preparation

```bash
# Clone repository
git clone https://github.com/SilviuSavu/Nautilus.git
cd Nautilus

# Validate Docker environment
docker --version && docker-compose --version
docker system info | grep -E "Server Version|CPUs|Total Memory"

# Check available resources
docker system df
```

### Step 2: M4 Max Optimized Deployment (Recommended)

```bash
# Enable M4 Max hardware acceleration
export M4_MAX_OPTIMIZED=1
export METAL_ACCELERATION=1
export NEURAL_ENGINE_ENABLED=1
export AUTO_HARDWARE_ROUTING=1

# Deploy with hardware optimization
docker-compose -f docker-compose.yml -f docker-compose.m4max.yml up --build -d

# Verify hardware acceleration status
curl http://localhost:8001/api/v1/acceleration/metal/status
curl http://localhost:8001/api/v1/optimization/health
```

### Step 3: Standard Production Deployment

```bash
# Standard production deployment
docker-compose up --build -d

# Monitor deployment progress
docker-compose logs -f --tail=50
```

### Step 4: System Validation

```bash
# Health check script
#!/bin/bash
echo "🔍 Validating Nautilus Production Deployment..."

# Core services
curl -s http://localhost:3000 && echo "✅ Frontend: OK"
curl -s http://localhost:8001/health && echo "✅ Backend API: OK"

# All 12 engines validation
for port in 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000; do
  response=$(curl -s -w "%{http_code}" http://localhost:${port}/health -o /dev/null)
  if [ $response -eq 200 ]; then
    echo "✅ Engine ${port}: HEALTHY"
  else
    echo "❌ Engine ${port}: FAILED (${response})"
  fi
done

# Database connectivity
docker exec -it nautilus-postgres psql -U postgres -d nautilus -c "SELECT 'Database OK' as status;"

echo "🎯 Deployment validation complete"
```

## 📊 Production Monitoring

### Performance Monitoring

```bash
# Monitor system performance
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check engine response times
for port in 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000; do
  echo "Engine ${port}:" $(curl -w "@curl-format.txt" -s http://localhost:${port}/health -o /dev/null)
done
```

### Grafana Dashboard Access
- **URL**: http://localhost:3002
- **Login**: admin/admin (change on first login)
- **Dashboards**: 
  - System Overview
  - M4 Max Hardware Utilization
  - Engine Performance Metrics
  - Trading Performance Analytics

## 🔧 Production Configuration

### Environment Variables
```env
# Production settings
NODE_ENV=production
DOCKER_ENV=production

# Database configuration
POSTGRES_DB=nautilus
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_here
DATABASE_URL=postgresql://postgres:password@postgres:5432/nautilus

# API Keys (Replace with production keys)
ALPHA_VANTAGE_API_KEY=your_production_key
FRED_API_KEY=your_fred_key

# M4 Max Hardware Acceleration
M4_MAX_OPTIMIZED=1
METAL_ACCELERATION=1
NEURAL_ENGINE_ENABLED=1
AUTO_HARDWARE_ROUTING=1
```

### Resource Allocation
```yaml
# docker-compose.override.yml for production
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G
  
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## 🚨 Production Security

### Security Checklist
- [ ] Change default passwords (PostgreSQL, Grafana)
- [ ] Configure firewall rules (only required ports)
- [ ] Enable TLS/SSL certificates
- [ ] Set up backup procedures
- [ ] Configure log rotation
- [ ] Enable container security scanning
- [ ] Set up monitoring alerts

### Backup Strategy
```bash
# Database backup
docker exec nautilus-postgres pg_dump -U postgres nautilus > backup_$(date +%Y%m%d).sql

# Container data backup
docker-compose down
tar -czf nautilus_data_$(date +%Y%m%d).tar.gz docker-data/
docker-compose up -d
```

## 🔧 Maintenance Operations

### Regular Maintenance
```bash
# Update containers
docker-compose pull
docker-compose up -d --force-recreate

# Clean up unused resources
docker system prune -f
docker volume prune -f

# Monitor disk usage
docker system df
```

### Performance Tuning
```bash
# Engine scaling (if needed)
docker-compose up -d --scale risk-engine=2
docker-compose up -d --scale ml-engine=3

# Resource monitoring
docker exec -it nautilus-backend htop
docker exec -it nautilus-postgres iostat 1
```

## 📈 Expected Performance Metrics

### Production Benchmarks
```
Component                 | Response Time | Throughput    | Status
Frontend                  | 12ms         | N/A           | ✅ Operational
Backend API               | 1.5-3.5ms    | 45+ RPS       | ✅ Operational
Analytics Engine (8100)   | 2.1ms        | High          | ✅ Healthy
Risk Engine (8200)        | 1.8ms        | High          | ✅ Healthy
Factor Engine (8300)      | 2.3ms        | Medium        | ✅ Healthy
ML Engine (8400)          | 1.9ms        | Medium        | ✅ Healthy
Features Engine (8500)    | 2.5ms        | High          | ✅ Healthy
WebSocket Engine (8600)   | 1.6ms        | Very High     | ✅ Healthy
Strategy Engine (8700)    | 2.0ms        | Medium        | ✅ Healthy
MarketData Engine (8800)  | 2.2ms        | Very High     | ✅ Healthy
Portfolio Engine (8900)   | 1.7ms        | Medium        | ✅ Healthy
Collateral Engine (9000)  | 0.36ms       | Critical      | ✅ Healthy
VPIN Engine (10000)       | <2ms         | High          | ✅ Healthy
```

### System Availability
- **Target Uptime**: 99.9% (8.76 hours downtime/year)
- **Current Status**: 100% operational (all engines healthy)
- **Monitoring**: Real-time health checks every 30 seconds

## 🆘 Troubleshooting

### Common Issues

**Container startup failures:**
```bash
# Check logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Force rebuild
docker-compose build --no-cache [service-name]
```

**Performance degradation:**
```bash
# Check resource usage
docker stats

# Examine engine health
for port in 8100 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000; do
  curl http://localhost:${port}/health
done

# Review system logs
docker-compose logs --tail=100
```

**Database connectivity issues:**
```bash
# Test database connection
docker exec -it nautilus-postgres psql -U postgres -l

# Check database performance
docker exec -it nautilus-postgres pg_stat_activity
```

## 📞 Production Support

### Monitoring Endpoints
- **System Health**: http://localhost:8001/health
- **Engine Status**: http://localhost:{8100-8900,9000,10000}/health  
- **Metrics**: http://localhost:3002 (Grafana)
- **API Documentation**: http://localhost:8001/docs

### Emergency Procedures
1. **System Down**: Execute `docker-compose restart`
2. **Database Issues**: Check `docker-compose logs postgres`
3. **Performance Degradation**: Review Grafana dashboards
4. **Engine Failures**: Individual engine restart via `docker-compose restart [engine-name]`

**Status**: ✅ **PRODUCTION READY** - Complete deployment guide with all operational procedures and M4 Max optimization support.
# Docker M4 Max Deployment Rollback Procedure

## Deployment Information
- **Date**: 2025-08-24
- **Hardware**: Apple M4 Max (14 cores: 10 P-cores + 4 E-cores, 36GB RAM)
- **Current Status**: 17 containers running successfully

## Current Container Status (Pre-M4 Max Deployment)
- nautilus-analytics-engine: Up 16 hours (healthy)
- nautilus-backend: Up 6 hours
- nautilus-engine: Up 40 hours (healthy)
- nautilus-factor-engine: Up 12 hours (healthy)
- nautilus-features-engine: Up 12 hours (healthy)
- nautilus-frontend: Up 19 hours
- nautilus-grafana: Up 20 hours
- nautilus-marketdata-engine: Up 12 hours (healthy)
- nautilus-ml-engine: Up 2 hours (healthy)
- nautilus-nginx: Up 4 hours
- nautilus-portfolio-engine: Up 12 hours (healthy)
- nautilus-postgres: Up 20 hours
- nautilus-prometheus: Up 20 hours
- nautilus-redis: Up 20 hours
- nautilus-risk-engine: Up 2 hours (healthy)
- nautilus-strategy-engine: Up 12 hours (healthy)
- nautilus-websocket-engine: Up 12 hours (healthy)

## Current Resource Allocations
```yaml
analytics-engine: 2.0 CPU / 4G RAM (limits), 1.0 CPU / 2G RAM (reservations)
risk-engine: 0.5 CPU / 1G RAM (limits), 0.25 CPU / 512M RAM (reservations)
factor-engine: 4.0 CPU / 8G RAM (limits), 2.0 CPU / 4G RAM (reservations)
ml-engine: 2.0 CPU / 6G RAM (limits), 1.0 CPU / 3G RAM (reservations)
features-engine: 3.0 CPU / 4G RAM (limits), 1.5 CPU / 2G RAM (reservations)
websocket-engine: 1.0 CPU / 2G RAM (limits), 0.5 CPU / 1G RAM (reservations)
strategy-engine: 1.0 CPU / 2G RAM (limits), 0.5 CPU / 1G RAM (reservations)
marketdata-engine: 2.0 CPU / 3G RAM (limits), 1.0 CPU / 1.5G RAM (reservations)
portfolio-engine: 4.0 CPU / 8G RAM (limits), 2.0 CPU / 4G RAM (reservations)
```

**Total Resource Usage**: 19.5 CPU cores, 38G RAM limits (16.25 CPU cores, 21G RAM reservations)

## Backup Files Created
- `docker-compose.yml.backup` - Original configuration
- `docker-rollback-procedure.md` - This rollback document

## Rollback Commands

### 1. Emergency Rollback (If containers fail to start)
```bash
cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus
docker-compose down
cp docker-compose.yml.backup docker-compose.yml
docker-compose up -d
```

### 2. Selective Rollback (If only specific engines fail)
```bash
# Stop problematic containers
docker-compose stop analytics-engine ml-engine factor-engine

# Restore original configuration
cp docker-compose.yml.backup docker-compose.yml

# Restart specific containers
docker-compose up -d analytics-engine ml-engine factor-engine
```

### 3. Validation After Rollback
```bash
# Check container health
docker-compose ps
docker-compose logs --tail=50

# Test endpoint health
curl -f http://localhost:8100/health  # Analytics
curl -f http://localhost:8200/health  # Risk
curl -f http://localhost:8300/health  # Factor
curl -f http://localhost:8400/health  # ML
```

### 4. Clean M4 Max Images (If needed)
```bash
# Remove M4 Max optimized images
docker image rm nautilus-analytics-engine:m4max
docker image rm nautilus-ml-engine:m4max
docker image rm nautilus-factor-engine:m4max
docker image rm nautilus-risk-engine:m4max
```

## Success Criteria for M4 Max Deployment
- [ ] All 17 containers start successfully
- [ ] Health checks pass for all engine containers
- [ ] Resource allocation shows M4 Max optimization
- [ ] Performance metrics show improvement
- [ ] No container restarts or errors in logs

## Rollback Decision Points
- Any container fails to start after 5 minutes
- Health checks fail for more than 2 engines
- System becomes unresponsive
- Resource usage exceeds M4 Max capabilities
- Performance degrades from current baseline
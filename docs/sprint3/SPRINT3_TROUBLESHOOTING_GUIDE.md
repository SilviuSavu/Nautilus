# Sprint 3 Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, diagnostic procedures, and solutions for Sprint 3 components including WebSocket infrastructure, risk management, analytics services, and strategy deployment.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [WebSocket Issues](#websocket-issues)
3. [Risk Management Issues](#risk-management-issues)
4. [Analytics Service Issues](#analytics-service-issues)
5. [Strategy Deployment Issues](#strategy-deployment-issues)
6. [Database Performance Issues](#database-performance-issues)
7. [Redis Connection Issues](#redis-connection-issues)
8. [Monitoring and Alerting Issues](#monitoring-and-alerting-issues)
9. [Performance Degradation](#performance-degradation)
10. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostic Checklist

### System Health Check Script

```bash
#!/bin/bash
# scripts/sprint3-health-check.sh

echo "=== Sprint 3 System Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check Docker containers
echo "1. Container Status:"
docker-compose ps
echo ""

# Check API endpoints
echo "2. API Health Checks:"
curl -s http://localhost:8001/health | jq '.'
curl -s http://localhost:8001/api/v1/sprint3/system/health | jq '.'
echo ""

# Check WebSocket connectivity
echo "3. WebSocket Connectivity:"
timeout 5s wscat -c ws://localhost:8001/ws/system/health -x '{"type":"ping"}' || echo "WebSocket connection failed"
echo ""

# Check database connectivity
echo "4. Database Status:"
docker exec nautilus-postgres pg_isready -U nautilus
echo ""

# Check Redis connectivity
echo "5. Redis Status:"
docker exec nautilus-redis redis-cli ping
echo ""

# Check Prometheus metrics
echo "6. Prometheus Metrics:"
curl -s http://localhost:9090/-/healthy || echo "Prometheus unhealthy"
echo ""

# Check resource usage
echo "7. Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Common Service Ports Check

```bash
#!/bin/bash
# Check if all required ports are accessible

ports=(3000 8001 5432 6379 9090 3001)
services=("Frontend" "Backend API" "PostgreSQL" "Redis" "Prometheus" "Grafana")

for i in "${!ports[@]}"; do
    port=${ports[$i]}
    service=${services[$i]}
    
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $service (port $port): Available"
    else
        echo "❌ $service (port $port): Not accessible"
    fi
done
```

---

## WebSocket Issues

### Issue: WebSocket Connections Failing

**Symptoms:**
- Frontend shows "Disconnected" status
- Real-time data not updating
- Console errors: "WebSocket connection failed"

**Diagnostic Steps:**

```bash
# 1. Check WebSocket service status
curl http://localhost:8001/api/v1/sprint3/websocket/stats

# 2. Test WebSocket connection manually
wscat -c ws://localhost:8001/ws/system/health

# 3. Check backend logs
docker-compose logs backend | grep -i websocket

# 4. Verify Redis pub/sub
docker exec nautilus-redis redis-cli monitor
```

**Common Solutions:**

1. **CORS Configuration Issue:**
```python
# backend/main.py - Ensure WebSocket origins are allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **WebSocket Manager Restart:**
```bash
# Restart WebSocket service
docker-compose restart backend

# Or restart specific WebSocket manager
curl -X POST http://localhost:8001/api/v1/sprint3/websocket/restart
```

3. **Connection Pool Exhaustion:**
```python
# Check connection pool status
import asyncio
from backend.websocket.websocket_manager import WebSocketManager

async def check_pool():
    manager = WebSocketManager()
    stats = await manager.get_connection_stats()
    print(f"Active connections: {stats['active']}")
    print(f"Max connections: {stats['max']}")

# Increase connection limits if needed
# backend/websocket/websocket_manager.py
MAX_CONNECTIONS = 2000  # Increase from default
```

### Issue: WebSocket Messages Not Received

**Symptoms:**
- WebSocket connection established but no messages received
- Subscription confirmations but no data updates

**Diagnostic Steps:**

```bash
# 1. Check subscription status
curl http://localhost:8001/api/v1/sprint3/websocket/subscriptions

# 2. Monitor Redis channels
docker exec nautilus-redis redis-cli psubscribe '*'

# 3. Test message broadcasting
curl -X POST http://localhost:8001/api/v1/sprint3/websocket/broadcast \
  -H "Content-Type: application/json" \
  -d '{"topic": "test", "message": {"test": true}}'
```

**Solutions:**

1. **Redis Channel Configuration:**
```python
# Verify Redis channel names match
REDIS_CHANNELS = {
    'portfolio_updates': 'portfolio:updates:*',
    'risk_alerts': 'risk:alerts:*',
    'analytics_updates': 'analytics:updates:*'
}
```

2. **Subscription Filter Debug:**
```javascript
// frontend/src/hooks/useWebSocketManager.ts
const subscribe = (topic, filters, callback) => {
  console.log('Subscribing to:', topic, 'with filters:', filters);
  // ... rest of subscription logic
};
```

### Issue: WebSocket Performance Degradation

**Symptoms:**
- Slow message delivery
- High memory usage in WebSocket service
- Connection timeouts

**Diagnostic Steps:**

```bash
# Check WebSocket metrics
curl http://localhost:8001/api/v1/sprint3/websocket/performance

# Monitor connection latency
ping localhost

# Check Redis memory usage
docker exec nautilus-redis redis-cli info memory
```

**Solutions:**

1. **Message Batching:**
```python
# backend/websocket/message_batching.py
class MessageBatcher:
    def __init__(self, batch_size=100, flush_interval=1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batches = {}
    
    async def add_message(self, topic, message):
        if topic not in self.batches:
            self.batches[topic] = []
        
        self.batches[topic].append(message)
        
        if len(self.batches[topic]) >= self.batch_size:
            await self.flush_batch(topic)
```

2. **Connection Cleanup:**
```python
# Implement proper connection cleanup
async def cleanup_stale_connections():
    stale_connections = []
    for conn_id, conn in connections.items():
        try:
            await conn.ping()
        except:
            stale_connections.append(conn_id)
    
    for conn_id in stale_connections:
        await remove_connection(conn_id)
```

---

## Risk Management Issues

### Issue: Risk Calculations Not Updating

**Symptoms:**
- VaR values stuck at old calculations
- Risk dashboard showing stale data
- No risk alerts being triggered

**Diagnostic Steps:**

```bash
# 1. Check risk service status
curl http://localhost:8001/api/v1/sprint3/risk/health

# 2. Verify risk monitoring is running
curl http://localhost:8001/api/v1/sprint3/risk/monitoring/status

# 3. Check for calculation errors
docker-compose logs backend | grep -i "risk\|var\|limit"

# 4. Test manual calculation
curl -X POST http://localhost:8001/api/v1/sprint3/analytics/risk/analyze \
  -H "Content-Type: application/json" \
  -d '{"portfolio_id": "PORTFOLIO_001"}'
```

**Solutions:**

1. **Restart Risk Monitoring:**
```bash
# Stop and restart risk monitoring
curl -X POST http://localhost:8001/api/v1/sprint3/risk/monitoring/stop
curl -X POST http://localhost:8001/api/v1/sprint3/risk/monitoring/start
```

2. **Clear Risk Cache:**
```python
# Clear cached risk calculations
import redis
r = redis.Redis(host='localhost', port=6379)
r.delete('risk:*')
```

3. **Database Index Optimization:**
```sql
-- Ensure proper indexes exist for risk queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_risk_timestamp 
ON risk_metrics (portfolio_id, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_risk_limits_active 
ON risk_limits (portfolio_id, active) WHERE active = true;
```

### Issue: Risk Limit Breaches Not Detected

**Symptoms:**
- Limits exceeded but no alerts sent
- Risk dashboard shows green despite breaches
- No email/Slack notifications

**Diagnostic Steps:**

```bash
# 1. Check limit monitoring status
curl http://localhost:8001/api/v1/sprint3/risk/limits/monitoring/status

# 2. Test breach detection manually
curl -X POST http://localhost:8001/api/v1/sprint3/risk/pre-trade-check \
  -H "Content-Type: application/json" \
  -d '{"portfolio_id": "PORTFOLIO_001", "trade": {"symbol": "AAPL", "quantity": 10000}}'

# 3. Check alert configuration
curl http://localhost:8001/api/v1/sprint3/risk/alerts/PORTFOLIO_001
```

**Solutions:**

1. **Enable Limit Monitoring:**
```python
# backend/risk/limit_monitor.py
async def start_monitoring():
    monitoring_enabled = await redis.get('risk_monitoring_enabled')
    if monitoring_enabled != 'true':
        await redis.set('risk_monitoring_enabled', 'true')
        logger.info("Risk monitoring enabled")
```

2. **Fix Alert Thresholds:**
```sql
-- Check and update alert thresholds
UPDATE risk_limits 
SET warning_threshold = threshold_value * 0.8 
WHERE warning_threshold IS NULL;
```

3. **Test Notification Channels:**
```python
# Test email notifications
from backend.notifications import email_service
await email_service.send_test_email("risk@company.com")

# Test Slack notifications
from backend.notifications import slack_service
await slack_service.send_test_message("#risk-alerts")
```

---

## Analytics Service Issues

### Issue: Performance Calculations Timeout

**Symptoms:**
- Analytics API returns 504 timeout
- Performance dashboard loading indefinitely
- High CPU usage on backend

**Diagnostic Steps:**

```bash
# 1. Check analytics service health
curl http://localhost:8001/api/v1/sprint3/analytics/health

# 2. Monitor resource usage
htop

# 3. Check for long-running queries
docker exec nautilus-postgres psql -U nautilus -d nautilus \
  -c "SELECT pid, query_start, query FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '30 seconds';"

# 4. Check Redis queue length
docker exec nautilus-redis redis-cli llen analytics_queue
```

**Solutions:**

1. **Optimize Analytics Queries:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_portfolio_returns_date 
ON portfolio_returns (portfolio_id, date DESC);

-- Use query timeout
SET statement_timeout = '30s';
```

2. **Implement Query Caching:**
```python
# backend/analytics/caching.py
from functools import wraps
import hashlib
import json

def cache_analytics_result(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = hashlib.md5(
                json.dumps([str(args), sorted(kwargs.items())]).encode()
            ).hexdigest()
            
            # Try cache first
            cached = await redis.get(f"analytics:{func.__name__}:{cache_key}")
            if cached:
                return json.loads(cached)
            
            # Calculate and cache
            result = await func(*args, **kwargs)
            await redis.setex(f"analytics:{func.__name__}:{cache_key}", ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator
```

3. **Use Background Processing:**
```python
# Offload heavy calculations to background tasks
from celery import Celery

app = Celery('analytics')

@app.task
def calculate_portfolio_performance(portfolio_id, start_date, end_date):
    # Heavy calculation logic here
    pass

# In API endpoint
task = calculate_portfolio_performance.delay(portfolio_id, start_date, end_date)
return {"task_id": task.id, "status": "processing"}
```

### Issue: Real-time Analytics Not Streaming

**Symptoms:**
- Analytics dashboard shows static data
- No real-time updates despite WebSocket connection
- Performance metrics not refreshing

**Diagnostic Steps:**

```bash
# 1. Check analytics streaming status
curl http://localhost:8001/api/v1/sprint3/analytics/streaming/status

# 2. Verify market data feed
curl http://localhost:8001/api/v1/market-data/status

# 3. Check analytics WebSocket messages
wscat -c ws://localhost:8001/ws/analytics/PORTFOLIO_001
```

**Solutions:**

1. **Restart Analytics Streaming:**
```python
# Restart the analytics streaming service
from backend.analytics.real_time_engine import RealTimeAnalyticsEngine

engine = RealTimeAnalyticsEngine()
await engine.restart()
```

2. **Check Data Pipeline:**
```python
# Verify data flow
async def check_data_pipeline():
    # Check if market data is flowing
    latest_data = await get_latest_market_data()
    if not latest_data:
        logger.error("No market data received")
    
    # Check if analytics calculations are running
    calc_status = await get_calculation_status()
    logger.info(f"Calculations status: {calc_status}")
```

---

## Strategy Deployment Issues

### Issue: Strategy Deployment Failures

**Symptoms:**
- Deployment stuck in "pending" status
- Strategy not appearing in production
- Rollback failures

**Diagnostic Steps:**

```bash
# 1. Check deployment status
curl http://localhost:8001/api/v1/sprint3/strategy/deployments

# 2. Check deployment logs
docker-compose logs strategy-deployer

# 3. Verify strategy configuration
curl http://localhost:8001/api/v1/sprint3/strategy/versions/STRATEGY_001
```

**Solutions:**

1. **Clear Stuck Deployments:**
```python
# backend/strategies/deployment_manager.py
async def clear_stuck_deployments():
    stuck_deployments = await db.query(
        "SELECT id FROM deployments WHERE status = 'pending' AND created_at < NOW() - INTERVAL '30 minutes'"
    )
    
    for deployment in stuck_deployments:
        await update_deployment_status(deployment.id, 'failed', 'Timeout')
```

2. **Manual Rollback:**
```bash
# Force rollback to previous version
curl -X POST http://localhost:8001/api/v1/sprint3/strategy/rollback \
  -H "Content-Type: application/json" \
  -d '{"deployment_id": "DEPLOY_001", "force": true}'
```

### Issue: Version Control Conflicts

**Symptoms:**
- Cannot create new strategy versions
- Git-like conflicts in strategy code
- Version history corruption

**Solutions:**

1. **Resolve Version Conflicts:**
```python
# backend/strategies/version_control.py
async def resolve_version_conflict(strategy_id, conflict_resolution='latest'):
    if conflict_resolution == 'latest':
        # Keep the latest version
        latest_version = await get_latest_version(strategy_id)
        await mark_as_canonical(latest_version.id)
    elif conflict_resolution == 'manual':
        # Require manual resolution
        await create_merge_request(strategy_id)
```

2. **Rebuild Version History:**
```python
# Rebuild corrupted version history
async def rebuild_version_history(strategy_id):
    versions = await get_all_versions(strategy_id)
    sorted_versions = sorted(versions, key=lambda v: v.created_at)
    
    for i, version in enumerate(sorted_versions):
        version.version_number = i + 1
        await db.commit()
```

---

## Database Performance Issues

### Issue: Slow Query Performance

**Symptoms:**
- API responses taking >5 seconds
- High database CPU usage
- Connection pool exhaustion

**Diagnostic Steps:**

```sql
-- Find slow queries
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Check active connections
SELECT count(*), state 
FROM pg_stat_activity 
GROUP BY state;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename IN ('portfolio_data', 'risk_metrics', 'analytics_data');
```

**Solutions:**

1. **Add Missing Indexes:**
```sql
-- Analytics performance indexes
CREATE INDEX CONCURRENTLY idx_analytics_portfolio_timestamp 
ON analytics_metrics (portfolio_id, timestamp DESC);

-- Risk monitoring indexes
CREATE INDEX CONCURRENTLY idx_risk_events_portfolio_type 
ON risk_events (portfolio_id, event_type, timestamp DESC);

-- WebSocket metrics indexes
CREATE INDEX CONCURRENTLY idx_websocket_metrics_timestamp 
ON websocket_metrics (timestamp DESC) 
WHERE timestamp > NOW() - INTERVAL '1 day';
```

2. **Optimize TimescaleDB:**
```sql
-- Add compression
SELECT add_compression_policy('analytics_metrics', INTERVAL '7 days');
SELECT add_compression_policy('risk_events', INTERVAL '3 days');

-- Add retention policy
SELECT add_retention_policy('websocket_metrics', INTERVAL '30 days');

-- Optimize chunk intervals
SELECT set_chunk_time_interval('analytics_metrics', INTERVAL '1 hour');
```

3. **Connection Pool Tuning:**
```python
# backend/database.py
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Increase from default 5
    max_overflow=30,       # Increase from default 10
    pool_pre_ping=True,    # Verify connections
    pool_recycle=3600      # Recycle connections after 1 hour
)
```

---

## Redis Connection Issues

### Issue: Redis Connection Timeouts

**Symptoms:**
- Intermittent Redis connection errors
- WebSocket messages not being published
- Cache misses despite recent writes

**Diagnostic Steps:**

```bash
# 1. Check Redis connectivity
docker exec nautilus-redis redis-cli ping

# 2. Monitor Redis connections
docker exec nautilus-redis redis-cli info clients

# 3. Check Redis memory usage
docker exec nautilus-redis redis-cli info memory

# 4. Monitor Redis slow log
docker exec nautilus-redis redis-cli slowlog get 10
```

**Solutions:**

1. **Configure Redis Connection Pool:**
```python
# backend/redis_config.py
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

pool = ConnectionPool(
    host='redis',
    port=6379,
    password=REDIS_PASSWORD,
    max_connections=50,
    retry_on_timeout=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

redis_client = redis.Redis(connection_pool=pool)
```

2. **Implement Redis Failover:**
```python
# backend/redis_failover.py
import asyncio

class RedisFailover:
    def __init__(self, primary_host, backup_hosts):
        self.primary_host = primary_host
        self.backup_hosts = backup_hosts
        self.current_client = None
    
    async def get_client(self):
        if self.current_client:
            try:
                await self.current_client.ping()
                return self.current_client
            except:
                pass
        
        # Try to connect to primary
        try:
            self.current_client = redis.Redis(host=self.primary_host)
            await self.current_client.ping()
            return self.current_client
        except:
            pass
        
        # Try backup hosts
        for backup_host in self.backup_hosts:
            try:
                self.current_client = redis.Redis(host=backup_host)
                await self.current_client.ping()
                return self.current_client
            except:
                continue
        
        raise ConnectionError("All Redis servers unavailable")
```

---

## Emergency Procedures

### Complete System Recovery

```bash
#!/bin/bash
# scripts/emergency-recovery.sh

echo "=== Emergency Recovery Procedure ==="

# 1. Stop all services
echo "Stopping all services..."
docker-compose down

# 2. Check disk space
echo "Checking disk space..."
df -h

# 3. Clean up Docker resources
echo "Cleaning up Docker resources..."
docker system prune -f
docker volume prune -f

# 4. Backup critical data
echo "Backing up critical data..."
docker run --rm -v nautilus_postgres_data:/source -v $(pwd)/backup:/backup alpine \
  tar czf /backup/postgres-emergency-$(date +%Y%m%d-%H%M%S).tar.gz -C /source .

# 5. Restart with minimal services
echo "Restarting with minimal services..."
docker-compose up -d postgres redis

# 6. Wait for database
echo "Waiting for database..."
sleep 30

# 7. Restart API server
echo "Starting API server..."
docker-compose up -d backend

# 8. Check system health
echo "Checking system health..."
sleep 30
curl -f http://localhost:8001/health || echo "API server not responding"

# 9. Restart frontend
echo "Starting frontend..."
docker-compose up -d frontend

echo "Emergency recovery completed. Please verify system functionality."
```

### Data Corruption Recovery

```sql
-- Check for data corruption
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Verify table integrity
VACUUM (VERBOSE, ANALYZE) portfolio_data;
VACUUM (VERBOSE, ANALYZE) risk_metrics;
VACUUM (VERBOSE, ANALYZE) analytics_metrics;

-- Rebuild corrupted indexes
REINDEX DATABASE nautilus;
```

### Performance Emergency Response

```bash
#!/bin/bash
# scripts/performance-emergency.sh

# 1. Identify resource-intensive processes
docker stats --no-stream
ps aux --sort=-%cpu | head -20

# 2. Scale down non-critical services
docker-compose scale analytics-worker=1
docker-compose scale websocket-handler=2

# 3. Enable aggressive caching
curl -X POST http://localhost:8001/api/v1/cache/enable-aggressive-mode

# 4. Increase connection limits temporarily
docker exec nautilus-postgres psql -U nautilus -c "ALTER SYSTEM SET max_connections = 500;"

# 5. Monitor improvement
while true; do
    echo "$(date): API Response Time: $(curl -w '%{time_total}' -s -o /dev/null http://localhost:8001/health)"
    sleep 10
done
```

### Service Isolation Procedure

When one component is failing and affecting others:

```bash
#!/bin/bash
# Isolate problematic service

SERVICE=$1

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service-name>"
    exit 1
fi

echo "Isolating service: $SERVICE"

# 1. Stop the problematic service
docker-compose stop $SERVICE

# 2. Check if other services recover
sleep 30
curl http://localhost:8001/health

# 3. Start service in debug mode
docker-compose run --rm $SERVICE /bin/bash

# 4. Manual investigation
echo "Service isolated. Investigate manually in the container shell."
```

This troubleshooting guide provides comprehensive solutions for the most common Sprint 3 issues. For issues not covered here, check the logs using `docker-compose logs [service]` and refer to the individual component documentation.
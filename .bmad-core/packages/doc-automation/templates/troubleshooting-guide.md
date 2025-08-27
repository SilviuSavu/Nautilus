# Troubleshooting Guide Template

**Status**: 📋 Template  
**Category**: Support Documentation  
**BMAD Package**: doc-automation v1.0.0

## Overview

**Brief troubleshooting guide description in bold text explaining common issues, diagnostic procedures, and support channels.**

## Quick Diagnostic Checklist

### System Status Check
- [ ] All services are running and healthy
- [ ] Database connections are active
- [ ] External API dependencies are responsive
- [ ] Sufficient disk space and memory available
- [ ] Network connectivity is stable
- [ ] SSL certificates are valid and not expired

### Health Check Commands
```bash
# Application health
curl -f http://localhost:8080/health || echo "Application unhealthy"

# Database connectivity
nc -zv localhost 5432 && echo "Database reachable" || echo "Database unreachable"

# Cache connectivity
redis-cli -h localhost -p 6379 ping || echo "Cache unreachable"

# System resources
df -h | grep -E "(Use%|tmp|var)"
free -h
top -bn1 | head -5
```

## Common Issues

### 🚨 Application Won't Start

#### Symptoms
- Container exits immediately after startup
- "Connection refused" errors
- Port binding failures
- Out of memory errors

#### Diagnostic Steps
```bash
# Check container status
docker-compose ps

# View application logs
docker-compose logs app --tail=50

# Check port availability
netstat -tulpn | grep :8080

# Verify environment configuration
docker-compose exec app env | grep -E "(DATABASE|REDIS|API)"

# Check resource usage
docker stats --no-stream
```

#### Common Causes & Solutions

##### Cause: Missing Environment Variables
**Error Message**: `Required environment variable DATABASE_URL is not set`

**Solution**:
```bash
# Check current environment
docker-compose exec app env | sort

# Verify .env file exists and is readable
ls -la .env
cat .env | grep -v "^#" | grep -v "^$"

# Restart with proper environment
docker-compose down
docker-compose up -d
```

##### Cause: Port Already in Use
**Error Message**: `Error starting userland proxy: listen tcp 0.0.0.0:8080: bind: address already in use`

**Solution**:
```bash
# Find process using the port
lsof -i :8080
netstat -tulpn | grep :8080

# Kill the process (if safe)
sudo kill -9 <PID>

# Or change the port in docker-compose.yml
# ports:
#   - "8081:8080"  # Use different external port
```

##### Cause: Database Connection Failure
**Error Message**: `FATAL: password authentication failed for user "app_user"`

**Solution**:
```bash
# Test database connection
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME}

# Reset database credentials
docker-compose down
docker volume rm $(docker volume ls -q | grep postgres)
docker-compose up -d

# Or fix credentials in .env file
# DATABASE_USER=correct_username
# DATABASE_PASSWORD=correct_password
```

### 🔌 Database Connection Issues

#### Symptoms
- "Connection timeout" errors
- "Too many connections" errors
- Slow query performance
- Data inconsistencies

#### Diagnostic Steps
```bash
# Check database container health
docker-compose logs database --tail=50

# Monitor active connections
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT 
    client_addr, 
    state, 
    query_start,
    LEFT(query, 50) as query
  FROM pg_stat_activity 
  WHERE state != 'idle' 
  ORDER BY query_start;
"

# Check database performance
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
  FROM pg_stat_user_tables 
  ORDER BY n_tup_ins DESC 
  LIMIT 10;
"
```

#### Solutions

##### Solution: Connection Pool Exhaustion
```bash
# Check current connection limits
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "SHOW max_connections;"

# Kill long-running queries
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT pg_terminate_backend(pid) 
  FROM pg_stat_activity 
  WHERE state != 'idle' 
    AND query_start < NOW() - INTERVAL '5 minutes';
"

# Optimize connection pooling in application
# Add to .env:
# DATABASE_MAX_CONNECTIONS=20
# DATABASE_IDLE_TIMEOUT=300
```

##### Solution: Database Performance Issues
```bash
# Update table statistics
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "ANALYZE;"

# Check for missing indexes
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT 
    schemaname, 
    tablename, 
    seq_scan, 
    seq_tup_read, 
    idx_scan, 
    idx_tup_fetch
  FROM pg_stat_user_tables 
  WHERE seq_scan > idx_scan 
  ORDER BY seq_tup_read DESC;
"

# Add indexes for frequently queried columns
# Example:
# CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
# CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at);
```

### 🌐 Network Connectivity Problems

#### Symptoms
- "Connection refused" from external services
- Intermittent API failures
- DNS resolution errors
- SSL/TLS handshake failures

#### Diagnostic Steps
```bash
# Test external connectivity
docker-compose exec app curl -I https://api.external-service.com/health
docker-compose exec app nslookup api.external-service.com
docker-compose exec app openssl s_client -connect api.external-service.com:443 -servername api.external-service.com

# Check internal network connectivity
docker network ls
docker network inspect $(docker-compose ps -q | head -1 | xargs docker inspect --format='{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}')

# Test container-to-container communication
docker-compose exec app nc -zv database 5432
docker-compose exec app nc -zv cache 6379
```

#### Solutions

##### Solution: DNS Resolution Issues
```bash
# Add custom DNS to containers
# In docker-compose.yml:
# services:
#   app:
#     dns:
#       - 8.8.8.8
#       - 1.1.1.1

# Or add to host file
docker-compose exec app sh -c 'echo "1.2.3.4 api.external-service.com" >> /etc/hosts'
```

##### Solution: SSL Certificate Problems
```bash
# Check certificate validity
openssl s_client -connect your-domain.com:443 -servername your-domain.com | grep -E "(subject|issuer|notAfter)"

# Renew Let's Encrypt certificates
sudo certbot renew --dry-run
sudo certbot renew

# Update certificate in container
docker-compose restart nginx
```

### 📊 Performance Issues

#### Symptoms
- Slow response times
- High CPU/Memory usage
- Request timeouts
- Queue backups

#### Diagnostic Steps
```bash
# Monitor system resources
top -bn1 | head -20
iostat -x 1 5
free -h && sync && echo 3 > /proc/sys/vm/drop_caches && free -h

# Monitor application performance
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Check application metrics (if available)
curl http://localhost:8080/metrics | grep -E "(response_time|request_rate|error_rate)"

# Monitor database performance
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;
"
```

#### Solutions

##### Solution: High Memory Usage
```bash
# Identify memory-intensive processes
docker stats --no-stream | sort -k4 -hr

# Add memory limits to containers
# In docker-compose.yml:
# services:
#   app:
#     mem_limit: 512m
#     memswap_limit: 512m

# Enable swap if needed (careful in production)
sudo swapon -s
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

##### Solution: High CPU Usage
```bash
# Profile application performance
docker-compose exec app top

# Enable CPU limits
# In docker-compose.yml:
# services:
#   app:
#     cpus: '1.5'
#     cpu_shares: 1024

# Scale horizontally if needed
docker-compose up -d --scale app=3
```

### 🔐 Authentication & Authorization Issues

#### Symptoms
- "Unauthorized" (401) errors
- "Forbidden" (403) errors
- Token validation failures
- Session timeouts

#### Diagnostic Steps
```bash
# Check authentication service
curl -I http://localhost:8080/auth/health

# Validate JWT tokens
# Use jwt.io or:
echo "YOUR_JWT_TOKEN" | cut -d. -f2 | base64 -d | jq .

# Check user permissions
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT u.username, r.role_name, p.permission_name
  FROM users u
  JOIN user_roles ur ON u.id = ur.user_id
  JOIN roles r ON ur.role_id = r.id
  JOIN role_permissions rp ON r.id = rp.role_id
  JOIN permissions p ON rp.permission_id = p.id
  WHERE u.username = 'problem_user';
"
```

#### Solutions

##### Solution: Token Expiration Issues
```bash
# Check token expiration settings
grep -r "TOKEN_EXPIRY\|JWT_EXPIRY" .env

# Implement token refresh mechanism
# Add refresh token endpoint
# Implement automatic token renewal in client

# Or extend token lifetime (temporary fix)
# JWT_EXPIRY=7200  # 2 hours instead of 1 hour
```

##### Solution: Permission Configuration Problems
```sql
-- Check user roles and permissions
SELECT u.username, r.role_name, p.permission_name
FROM users u
JOIN user_roles ur ON u.id = ur.user_id
JOIN roles r ON ur.role_id = r.id
JOIN role_permissions rp ON r.id = rp.role_id
JOIN permissions p ON rp.permission_id = p.id
WHERE u.username = 'problem_user';

-- Grant additional permissions if needed
INSERT INTO role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM roles r, permissions p
WHERE r.role_name = 'user' AND p.permission_name = 'read_data';
```

## Error Code Reference

### HTTP Status Codes
| Code | Meaning | Common Causes | Solutions |
|------|---------|---------------|-----------|
| 400 | Bad Request | Invalid JSON, missing parameters | Validate request format |
| 401 | Unauthorized | Invalid/missing auth token | Check authentication |
| 403 | Forbidden | Insufficient permissions | Verify user roles |
| 404 | Not Found | Invalid endpoint/resource | Check URL and existence |
| 422 | Unprocessable Entity | Validation errors | Fix data validation issues |
| 429 | Too Many Requests | Rate limit exceeded | Implement backoff/retry |
| 500 | Internal Server Error | Application/database error | Check logs and fix code |
| 502 | Bad Gateway | Upstream service down | Check service dependencies |
| 503 | Service Unavailable | Service overloaded | Scale or restart services |
| 504 | Gateway Timeout | Upstream timeout | Increase timeout/optimize |

### Application-Specific Error Codes
| Code | Message | Cause | Solution |
|------|---------|--------|----------|
| APP_001 | Database connection failed | DB server down/unreachable | Check database status |
| APP_002 | External API timeout | Third-party service slow | Increase timeout/retry |
| APP_003 | Invalid configuration | Missing/wrong config values | Verify environment variables |
| APP_004 | Rate limit exceeded | Too many requests | Implement request throttling |
| APP_005 | File upload failed | Storage issues/limits | Check disk space/permissions |

## Monitoring & Alerting

### Key Metrics to Monitor
```yaml
application_metrics:
  response_time:
    warning_threshold: 500ms
    critical_threshold: 2000ms
  
  error_rate:
    warning_threshold: 1%
    critical_threshold: 5%
  
  request_rate:
    normal_range: 10-100 RPS
    alert_on_deviation: 50%

system_metrics:
  cpu_usage:
    warning_threshold: 70%
    critical_threshold: 90%
  
  memory_usage:
    warning_threshold: 80%
    critical_threshold: 95%
  
  disk_usage:
    warning_threshold: 80%
    critical_threshold: 95%

business_metrics:
  user_registrations:
    expected_rate: 10-50/hour
    alert_on_zero: 30min
  
  transaction_success_rate:
    minimum_threshold: 95%
    alert_immediately: true
```

### Alert Configuration
```yaml
alerts:
  - name: "Application Down"
    condition: "http_status != 200"
    duration: "2m"
    severity: "critical"
    channels: ["email", "slack", "pagerduty"]
  
  - name: "High Error Rate"
    condition: "error_rate > 5%"
    duration: "5m"
    severity: "warning"
    channels: ["email", "slack"]
  
  - name: "Database Connection Issues"
    condition: "db_connection_errors > 0"
    duration: "1m"
    severity: "critical"
    channels: ["email", "slack", "pagerduty"]
```

## Support Escalation

### Severity Levels
| Level | Response Time | Description | Examples |
|-------|---------------|-------------|----------|
| **P0 - Critical** | 15 minutes | Complete service outage | App completely down, data loss |
| **P1 - High** | 1 hour | Major functionality impacted | Login system down, payment failures |
| **P2 - Medium** | 4 hours | Minor functionality issues | UI bugs, slow performance |
| **P3 - Low** | 1 business day | Cosmetic or enhancement requests | Documentation updates, feature requests |

### Contact Information
- **On-call Engineering**: +1-555-ON-CALL (662255)
- **Support Email**: support@your-company.com
- **Status Page**: https://status.your-company.com
- **Slack Channel**: #incident-response
- **PagerDuty**: https://your-company.pagerduty.com

### Escalation Procedure
1. **Initial Response** (15 min): Acknowledge incident, assess severity
2. **Investigation** (30 min): Diagnose root cause, gather logs
3. **Communication** (45 min): Update stakeholders with findings
4. **Resolution** (varies): Implement fix, verify solution
5. **Post-mortem** (24 hours): Document lessons learned

## Recovery Procedures

### Service Recovery Checklist
- [ ] Identify and isolate the problem
- [ ] Stop the affected service if necessary
- [ ] Restore from backup if data corruption occurred
- [ ] Apply the fix or rollback to previous version
- [ ] Restart services in proper dependency order
- [ ] Verify all services are healthy
- [ ] Monitor for recurring issues
- [ ] Update incident documentation

### Rollback Procedures
```bash
# Quick rollback to previous version
docker-compose down
docker-compose pull your-app:previous-version
docker-compose up -d

# Database rollback (if schema changes)
docker-compose exec -T database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} < backup/pre-deployment.sql

# Configuration rollback
cp backup/.env.backup .env
docker-compose restart

# Verify rollback success
curl https://your-domain.com/health
./run-smoke-tests.sh
```

### Data Recovery
```bash
# Restore from latest backup
./restore-backup.sh $(ls -t backups/ | head -1)

# Verify data integrity
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "
  SELECT COUNT(*) FROM users;
  SELECT COUNT(*) FROM transactions;
  SELECT MAX(created_at) FROM audit_log;
"

# Run data validation checks
./validate-data-integrity.sh
```

## Prevention Strategies

### Monitoring Improvements
- Implement comprehensive health checks
- Set up proactive alerting
- Use distributed tracing for complex requests
- Monitor business metrics, not just technical ones
- Regular synthetic transaction testing

### Code Quality
- Increase test coverage
- Implement code review processes
- Use static code analysis tools
- Regular security vulnerability scans
- Performance testing in staging environment

### Infrastructure Resilience
- Implement circuit breakers for external services
- Use connection pooling and retry mechanisms
- Set appropriate timeout values
- Plan for graceful service degradation
- Regular disaster recovery testing

## Documentation Updates

When resolving issues, update this troubleshooting guide:

1. **Add New Issues**: Document new problems and solutions
2. **Update Procedures**: Revise diagnostic steps based on experience
3. **Improve Monitoring**: Add new metrics based on discovered blind spots
4. **Refine Alerts**: Adjust thresholds based on false positive rates
5. **Share Knowledge**: Conduct post-incident reviews and share learnings

---

**Document Information**:
- **Version**: 1.0.0
- **Last Updated**: $(date)
- **Next Review**: Monthly
- **Incident Count**: 0 (track resolved incidents)

**Generated by**: BMAD Documentation Template System  
**Template**: troubleshooting-guide.md

## Template Usage

This template should be customized by:
1. Adding application-specific error codes and messages
2. Including actual monitoring thresholds and metrics
3. Updating contact information and escalation procedures
4. Adding real diagnostic commands for your stack
5. Including project-specific recovery procedures

### BMAD Commands for Troubleshooting Documentation

```bash
# Apply this template to new troubleshooting guide
bmad apply template troubleshooting-guide target=docs/support/new-service-troubleshooting.md

# Validate troubleshooting documentation standards
bmad run check-doc-health include_patterns="['docs/support/**']"

# Generate troubleshooting cross-references
bmad run generate-doc-sitemap include_patterns="['docs/support/**']" group_by=category
```
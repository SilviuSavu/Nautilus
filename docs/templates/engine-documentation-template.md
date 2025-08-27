# [Engine Name] Documentation Template

**[Brief description of engine purpose and functionality]**

## 🎯 Overview

**Port**: [Port Number]  
**Status**: ✅ **[Current Status]**  
**Performance**: [Response Time] average response time  
**Integration**: [Integration details]

### Key Capabilities
- **[Capability 1]**: [Description]
- **[Capability 2]**: [Description]
- **[Capability 3]**: [Description]
- **[Hardware Acceleration]**: [M4 Max optimization details if applicable]

## 🔧 Technical Specifications

### Architecture
- **Framework**: [Technology stack]
- **Dependencies**: [Key dependencies]
- **Data Storage**: [Storage requirements]
- **Memory Usage**: [Memory footprint]
- **CPU Usage**: [CPU requirements]

### Performance Metrics
```
Metric                    | Value           | Status
Response Time             | [X]ms           | ✅ Optimal
Throughput               | [X] ops/sec     | ✅ High
Memory Usage             | [X]MB           | ✅ Efficient
CPU Utilization          | [X]%            | ✅ Optimized
```

## 📡 API Reference

### Health Check
```bash
GET /health
# Returns: {"status": "healthy", "timestamp": "...", "engine": "[engine-name]"}
```

### Core Endpoints
```bash
# [Endpoint 1]
[HTTP METHOD] /api/v1/[endpoint-path]
# Description: [What this endpoint does]
# Parameters: [Required parameters]
# Response: [Expected response format]

# [Endpoint 2]
[HTTP METHOD] /api/v1/[endpoint-path]
# Description: [What this endpoint does]
# Parameters: [Required parameters]
# Response: [Expected response format]
```

### WebSocket Endpoints (if applicable)
```bash
WS /ws/[websocket-path]
# Description: [WebSocket functionality]
# Events: [Event types sent/received]
```

## 🔄 Integration Points

### MessageBus Integration
- **Subscribes to**: [Message types this engine subscribes to]
- **Publishes**: [Message types this engine publishes]
- **Message Format**: [Standard message structure]

### Database Integration
- **Tables Used**: [Database tables accessed]
- **Operations**: [Read/Write operations performed]
- **Caching**: [Caching strategy if applicable]

### External APIs
- **[External Service 1]**: [Integration details]
- **[External Service 2]**: [Integration details]

## 🚀 Deployment Configuration

### Docker Configuration
```yaml
# docker-compose.yml section for this engine
[engine-name]:
  build:
    context: ./backend/engines/[engine-folder]
    dockerfile: Dockerfile
  ports:
    - "[port]:[port]"
  environment:
    - ENGINE_PORT=[port]
    - [OTHER_ENV_VARS]=value
  depends_on:
    - postgres
    - redis
```

### Environment Variables
```env
# Required environment variables
[ENGINE_NAME]_PORT=[port]
[ENGINE_NAME]_DEBUG=false
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# Optional configuration
[ENGINE_NAME]_WORKERS=4
[ENGINE_NAME]_TIMEOUT=30
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Basic health check
curl http://localhost:[port]/health

# Detailed status
curl http://localhost:[port]/status

# Metrics endpoint
curl http://localhost:[port]/metrics
```

### Logging
- **Log Level**: [Default log level]
- **Log Format**: [JSON/Plain text]
- **Key Log Events**: [Important events logged]

### Grafana Dashboards
- **Dashboard Name**: [Engine Name] Performance
- **Key Metrics**: [Metrics displayed]
- **Alerts**: [Alert conditions configured]

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
cd backend/engines/[engine-folder]
python -m pytest tests/unit/

# Test coverage
python -m pytest --cov=[engine-module] tests/
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/test_[engine-name].py

# End-to-end testing
curl -X POST http://localhost:[port]/api/v1/test
```

### Performance Tests
```bash
# Load testing
ab -n 1000 -c 10 http://localhost:[port]/health

# Stress testing
docker run --rm -i loadimpact/k6 run - <test-script.js
```

## 🔧 Development

### Local Development
```bash
# Start engine locally
cd backend/engines/[engine-folder]
python main.py

# Development with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port [port]
```

### Configuration
- **Config File**: [Location of configuration files]
- **Key Settings**: [Important configuration options]
- **Environment Override**: [How to override config with environment variables]

## 🚨 Troubleshooting

### Common Issues

**[Issue 1]**: [Description]
```bash
# Solution commands
[command to resolve]
```

**[Issue 2]**: [Description]
```bash
# Solution commands
[command to resolve]
```

### Debug Mode
```bash
# Enable debug logging
export [ENGINE_NAME]_DEBUG=true
docker-compose restart [engine-name]

# View debug logs
docker-compose logs -f [engine-name]
```

### Performance Issues
```bash
# Check resource usage
docker stats [container-name]

# Profile engine performance
curl http://localhost:[port]/debug/profile
```

## 📋 Maintenance

### Regular Maintenance
- **Log Rotation**: [How logs are rotated]
- **Cache Cleanup**: [Cache maintenance procedures]
- **Database Maintenance**: [Database cleanup tasks]

### Updates
```bash
# Update engine
docker-compose pull [engine-name]
docker-compose up -d [engine-name]

# Rolling update (zero-downtime)
[rolling update procedure]
```

### Backup & Recovery
- **Data Backup**: [What data needs backing up]
- **Recovery Procedures**: [How to restore from backup]

## 📚 References

### Documentation Links
- **[Related Engine 1]**: [Link to related engine documentation]
- **[Related Engine 2]**: [Link to related engine documentation]
- **[External API Docs]**: [Link to external API documentation]

### Code Repository
- **Location**: `backend/engines/[engine-folder]/`
- **Key Files**: [Important source files]
- **Dependencies**: `requirements.txt`

---

**Last Updated**: [Date]  
**Version**: [Engine Version]  
**Status**: ✅ **[Current Status]** - [Brief status description]
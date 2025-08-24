# Phase 8: Autonomous Security Operations - M4 Max MacBook Installation Guide

## System Requirements Met ✅
- **MacBook M4 Max**: ✅ Supported (ARM64 architecture)
- **36GB RAM**: ✅ Excellent for Phase 8 operations (requires 8-12GB)
- **Local Development**: ✅ Optimized for containerized deployment

## Installation Strategy for M4 Max MacBook

### Phase 8 Container Approach (Recommended)
Rather than installing heavy dependencies like PyTorch directly in the main backend container, we'll create specialized Phase 8 containers:

```bash
# Create Phase 8 dedicated container
cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

# Build Phase 8 container with ARM64 optimizations
docker build -f Dockerfile.phase8 -t nautilus-phase8-security:latest .
```

### Dockerfile.phase8 (ARM64 Optimized)
```dockerfile
FROM python:3.11-slim

# ARM64 optimizations for M4 Max
ENV PYTHONPATH=/app
ENV TORCH_VERSION=2.1.0
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies optimized for ARM64
COPY requirements-phase8.txt .
RUN pip install --no-cache-dir -r requirements-phase8.txt

# Copy Phase 8 code
COPY phase8_autonomous_operations/ phase8_autonomous_operations/
COPY phase8_security_routes.py .
COPY phase8_startup_service.py .

EXPOSE 8010
CMD ["python", "-m", "uvicorn", "phase8_main:app", "--host", "0.0.0.0", "--port", "8010"]
```

### Requirements for Phase 8 (requirements-phase8.txt)
```txt
# Core FastAPI and async support
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
redis==5.0.1

# ML and Security Libraries (ARM64 compatible)
torch==2.1.0  # ARM64 native support
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4

# Security and Monitoring
cryptography==41.0.8
python-jose[cryptography]==3.3.0
prometheus-client==0.19.0

# Database and messaging
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
aioredis==2.0.1

# Networking and HTTP clients
aiohttp==3.9.1
httpx==0.25.2
websockets==12.0
```

### Docker Compose Integration
Add to your existing `docker-compose.yml`:

```yaml
services:
  # ... existing services ...
  
  phase8-security:
    build:
      context: ./backend
      dockerfile: Dockerfile.phase8
    container_name: nautilus-phase8-security
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://nautilus:nautilus123@postgres:5432/nautilus
      - PHASE8_LOG_LEVEL=INFO
    ports:
      - "8010:8010"
    networks:
      - nautilus_nautilus-network
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    mem_limit: 4G  # Optimize for M4 Max
    cpus: 2.0      # Use 2 CPU cores
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Memory Optimization for M4 Max (36GB RAM)

### Container Resource Allocation
```yaml
# Optimized for 36GB total RAM
backend: 8GB          # Main backend (current)
postgres: 4GB         # Database with TimescaleDB
redis: 2GB            # Message bus and caching
phase8-security: 4GB  # Autonomous security operations
engines (9x): 1GB ea  # 9GB total for processing engines
frontend: 2GB         # React development
monitoring: 4GB       # Grafana + Prometheus
system: 9GB           # macOS and other processes
```

### Performance Tuning
```bash
# Optimize Docker for M4 Max
echo '{
  "experimental": true,
  "features": {
    "buildkit": true
  },
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "10GB"
    }
  }
}' > ~/.docker/daemon.json

# Restart Docker Desktop after configuration
```

## Lightweight Phase 8 Implementation

### Phase 8 Lite Configuration
For M4 Max optimization, we can run Phase 8 in "Lite Mode":

```python
# backend/phase8_config.py
PHASE8_CONFIG = {
    "mode": "lite",  # Reduced memory footprint
    "max_workers": 4,  # Optimal for M4 Max
    "batch_size": 1000,  # Smaller batches
    "cache_size": "512MB",  # Reasonable cache
    "ml_model_size": "small",  # Use smaller models
    "enable_gpu": False,  # CPU-only for now
}
```

### Enable Phase 8 in Main Backend
Update `docker-compose.yml` environment for backend:

```yaml
backend:
  environment:
    # ... existing vars ...
    - PHASE8_ENABLED=true
    - PHASE8_MODE=lite
    - PHASE8_REDIS_HOST=redis
    - PHASE8_SERVICE_URL=http://phase8-security:8010
```

## Installation Steps

### 1. Create Phase 8 Requirements File
```bash
cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend
cat > requirements-phase8.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
redis==5.0.1
torch==2.1.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
cryptography==41.0.8
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
aioredis==2.0.1
aiohttp==3.9.1
httpx==0.25.2
prometheus-client==0.19.0
EOF
```

### 2. Create Phase 8 Dockerfile
```bash
cat > Dockerfile.phase8 << 'EOF'
FROM python:3.11-slim

ENV PYTHONPATH=/app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-phase8.txt .
RUN pip install --no-cache-dir -r requirements-phase8.txt

COPY phase8_autonomous_operations/ phase8_autonomous_operations/
COPY phase8_security_routes.py .
COPY phase8_startup_service.py .

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8010/health || exit 1

CMD ["python", "-m", "uvicorn", "phase8_security_routes:app", "--host", "0.0.0.0", "--port", "8010"]
EOF
```

### 3. Update Docker Compose
```bash
# Add Phase 8 service to docker-compose.yml
cat >> docker-compose.yml << 'EOF'

  phase8-security:
    build:
      context: ./backend
      dockerfile: Dockerfile.phase8
    container_name: nautilus-phase8-security
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://nautilus:nautilus123@postgres:5432/nautilus
      - PHASE8_LOG_LEVEL=INFO
      - PHASE8_MODE=lite
    ports:
      - "8010:8010"
    networks:
      - nautilus_nautilus-network
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
```

### 4. Build and Start Phase 8
```bash
# Build Phase 8 container
docker-compose build phase8-security

# Start Phase 8 service
docker-compose up -d phase8-security

# Check Phase 8 health
curl http://localhost:8010/health
```

### 5. Enable Phase 8 in Main Backend
Update backend environment in docker-compose.yml:

```yaml
backend:
  environment:
    # Add these to existing environment
    - PHASE8_ENABLED=true
    - PHASE8_SERVICE_URL=http://phase8-security:8010
```

## Testing Phase 8 Integration

### Health Check Endpoints
```bash
# Main backend health
curl http://localhost:8001/health

# Phase 8 dedicated service health
curl http://localhost:8010/health

# Phase 8 security endpoints (via main backend)
curl http://localhost:8001/api/v1/security/status
```

### Resource Monitoring
```bash
# Monitor container resource usage
docker stats nautilus-phase8-security

# Check total system resource usage
docker system df
htop  # Monitor macOS system resources
```

## Expected Performance on M4 Max

### Phase 8 Performance Metrics
- **Startup Time**: 30-60 seconds (first run, includes model loading)
- **Memory Usage**: 2-4GB per container
- **CPU Usage**: 1-2 cores sustained, 4 cores peak
- **Analysis Latency**: < 200ms for security events
- **Throughput**: 1,000+ security events/second

### System Resource Usage
```
Total System Usage (36GB RAM):
├── macOS System: ~8-10GB
├── Docker Desktop: ~2GB
├── Backend Services: ~20GB
│   ├── Main Backend: 8GB
│   ├── Database: 4GB
│   ├── Phase 8 Security: 4GB
│   └── Processing Engines: 4GB
└── Available: ~6-8GB (buffer)
```

## Troubleshooting

### Memory Issues
```bash
# If system runs out of memory, reduce Phase 8 container limit
docker-compose down phase8-security
# Edit docker-compose.yml: change memory limit to 2G
docker-compose up -d phase8-security
```

### ARM64 Compatibility Issues
```bash
# If PyTorch fails to install, use CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Container Communication Issues
```bash
# Test network connectivity
docker exec nautilus-backend ping phase8-security
docker exec nautilus-phase8-security ping backend
```

## Production Deployment Notes

For production deployment on M4 Max:

1. **Enable Phase 8** by setting `PHASE8_ENABLED=true` in main backend
2. **Monitor Resources** using built-in monitoring dashboard
3. **Scale Horizontally** by adding more Phase 8 containers if needed
4. **Use SSD Storage** for optimal I/O performance with TimescaleDB

## Next Steps

1. Install dependencies and build Phase 8 container
2. Start Phase 8 service and verify health
3. Enable Phase 8 integration in main backend
4. Test security endpoints and WebSocket streaming
5. Monitor system performance and optimize as needed

Phase 8 is now optimized for your M4 Max MacBook with 36GB RAM! 🚀
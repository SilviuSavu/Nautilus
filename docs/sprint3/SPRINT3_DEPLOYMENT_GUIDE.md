# Sprint 3 Deployment Guide

## Overview

This comprehensive guide covers deploying Sprint 3 advanced trading infrastructure across development, staging, and production environments with complete monitoring, observability, and scaling capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Infrastructure Components](#infrastructure-components)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Configuration](#security-configuration)
8. [Scaling & Performance](#scaling--performance)
9. [Health Checks & Monitoring](#health-checks--monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements (Production)
- **CPU**: 16 cores
- **RAM**: 32GB
- **Storage**: 200GB NVMe SSD
- **Network**: 1 Gbps

### Software Dependencies

```bash
# Required software versions
Docker >= 20.10.0
Docker Compose >= 2.0.0
Node.js >= 18.0.0
Python >= 3.11
PostgreSQL >= 14.0
Redis >= 6.2.0
Prometheus >= 2.35.0
Grafana >= 9.0.0
```

### API Keys and Credentials

Create a `.env` file with required credentials:

```bash
# Core API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
DATAGOV_API_KEY=your_datagov_key
TRADING_ECONOMICS_API_KEY=your_trading_economics_key

# Database Configuration
POSTGRES_DB=nautilus
POSTGRES_USER=nautilus
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://nautilus:password@postgres:5432/nautilus

# Redis Configuration
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=your_redis_password

# Security
JWT_SECRET_KEY=your_jwt_secret_key
API_SECRET_KEY=your_api_secret_key
ENCRYPTION_KEY=your_32_character_encryption_key

# Interactive Brokers
IB_GATEWAY_HOST=ib-gateway
IB_GATEWAY_PORT=4001
IB_GATEWAY_CLIENT_ID=1

# WebSocket Configuration
WEBSOCKET_REDIS_CHANNEL=nautilus_websocket
WEBSOCKET_MAX_CONNECTIONS=1000

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_ADMIN_PASSWORD=your_grafana_password
```

---

## Environment Setup

### Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/nautilus.git
cd nautilus

# Create environment file
cp .env.example .env.development

# Build and start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Staging Environment

```bash
# Create staging configuration
cp .env.example .env.staging

# Configure staging-specific settings
export ENVIRONMENT=staging
export LOG_LEVEL=INFO

# Deploy staging environment
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Production Environment

```bash
# Create production configuration
cp .env.example .env.production

# Configure production settings
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export ENABLE_MONITORING=true

# Deploy production environment
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

---

## Infrastructure Components

### Core Services Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Frontend Application
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.production
    ports:
      - "3000:80"
    environment:
      - VITE_API_BASE_URL=${API_BASE_URL}
      - VITE_WS_URL=${WEBSOCKET_URL}
    depends_on:
      - backend
    networks:
      - nautilus-network

  # Backend API Service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.production
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - ENVIRONMENT=${ENVIRONMENT}
    depends_on:
      - postgres
      - redis
    networks:
      - nautilus-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database with TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./schema/sql:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - nautilus-network

  # Redis for Caching and Pub/Sub
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - nautilus-network

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - nautilus-network

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    networks:
      - nautilus-network

networks:
  nautilus-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## Docker Deployment

### Build Process

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build backend

# Build with no cache
docker-compose build --no-cache
```

### Multi-Stage Production Dockerfile

#### Backend Dockerfile.production
```dockerfile
# Multi-stage build for production backend
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r nautilus && useradd -r -g nautilus nautilus
RUN chown -R nautilus:nautilus /app
USER nautilus

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Frontend Dockerfile.production
```dockerfile
# Multi-stage build for production frontend
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage with nginx
FROM nginx:alpine as production

# Copy built application
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.production.conf /etc/nginx/conf.d/default.conf

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Deployment Scripts

#### deploy.sh
```bash
#!/bin/bash

set -e

ENVIRONMENT=${1:-development}
CONFIG_FILE="docker-compose.${ENVIRONMENT}.yml"

echo "Deploying to ${ENVIRONMENT} environment..."

# Validate environment
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Load environment variables
if [[ -f ".env.${ENVIRONMENT}" ]]; then
    source ".env.${ENVIRONMENT}"
fi

# Pre-deployment checks
echo "Running pre-deployment checks..."
docker-compose -f docker-compose.yml -f "$CONFIG_FILE" config > /dev/null

# Build and deploy
echo "Building services..."
docker-compose -f docker-compose.yml -f "$CONFIG_FILE" build

echo "Starting services..."
docker-compose -f docker-compose.yml -f "$CONFIG_FILE" up -d

# Health checks
echo "Waiting for services to be healthy..."
sleep 30

# Verify deployment
echo "Verifying deployment..."
./scripts/health-check.sh

echo "Deployment to ${ENVIRONMENT} completed successfully!"
```

---

## Kubernetes Deployment

### Namespace Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nautilus-sprint3
  labels:
    name: nautilus-sprint3
    environment: production
```

### ConfigMap for Environment Variables

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nautilus-config
  namespace: nautilus-sprint3
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_HOST: "postgres-service"
  REDIS_HOST: "redis-service"
  PROMETHEUS_URL: "http://prometheus-service:9090"
```

### Backend Deployment

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nautilus-backend
  namespace: nautilus-sprint3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nautilus-backend
  template:
    metadata:
      labels:
        app: nautilus-backend
    spec:
      containers:
      - name: backend
        image: nautilus/backend:sprint3-latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nautilus-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: nautilus-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: nautilus-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nautilus-backend-service
  namespace: nautilus-sprint3
spec:
  selector:
    app: nautilus-backend
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001
  type: ClusterIP
```

### Database StatefulSet

```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: nautilus-sprint3
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: timescale/timescaledb:latest-pg14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "nautilus"
        - name: POSTGRES_USER
          value: "nautilus"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: nautilus-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: init-scripts
          mountPath: /docker-entrypoint-initdb.d
      volumes:
      - name: init-scripts
        configMap:
          name: postgres-init-scripts
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nautilus-ingress
  namespace: nautilus-sprint3
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.nautilus.com
    - app.nautilus.com
    secretName: nautilus-tls
  rules:
  - host: api.nautilus.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nautilus-backend-service
            port:
              number: 8001
  - host: app.nautilus.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nautilus-frontend-service
            port:
              number: 80
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'nautilus-backend'
    static_configs:
      - targets: ['backend:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: nautilus-sprint3-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for {{ $labels.instance }}"

      - alert: WebSocketConnectionsDrop
        expr: websocket_active_connections < 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "WebSocket connections dropped significantly"

      - alert: RiskLimitBreach
        expr: risk_limit_utilization > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Risk limit breach detected"
          description: "Risk limit utilization is {{ $value }}"

      - alert: DatabaseConnectionPoolExhausted
        expr: postgres_connection_pool_active / postgres_connection_pool_max > 0.9
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Nautilus Sprint 3 Trading Overview",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "WebSocket Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "websocket_active_connections",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Risk Metrics",
        "type": "table",
        "targets": [
          {
            "expr": "risk_var_95",
            "legendFormat": "VaR 95%"
          }
        ]
      }
    ]
  }
}
```

---

## Security Configuration

### SSL/TLS Setup

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nautilus.key \
    -out nautilus.crt \
    -subj "/CN=nautilus.com"

# Create Kubernetes secret
kubectl create secret tls nautilus-tls \
    --cert=nautilus.crt \
    --key=nautilus.key \
    --namespace=nautilus-sprint3
```

### Secrets Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nautilus-secrets
  namespace: nautilus-sprint3
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  jwt-secret: <base64-encoded-jwt-secret>
  api-secret: <base64-encoded-api-secret>
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nautilus-network-policy
  namespace: nautilus-sprint3
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nautilus-sprint3
    ports:
    - protocol: TCP
      port: 8001
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: nautilus-sprint3
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
```

---

## Scaling & Performance

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nautilus-backend-hpa
  namespace: nautilus-sprint3
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nautilus-backend
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: websocket_connections_per_pod
      target:
        type: AverageValue
        averageValue: "100"
```

### Load Testing

```bash
# Load testing script
#!/bin/bash

# Install k6 if not already installed
if ! command -v k6 &> /dev/null; then
    echo "Installing k6..."
    brew install k6  # macOS
    # or: sudo apt-get install k6  # Ubuntu
fi

# Run load test
k6 run --vus 100 --duration 30s load-test.js
```

```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export default function () {
  // Test API endpoints
  const apiResponse = http.get('http://api.nautilus.com/health');
  check(apiResponse, {
    'API status is 200': (r) => r.status === 200,
    'API response time < 500ms': (r) => r.timings.duration < 500,
  });

  // Test WebSocket connections
  const wsResponse = http.get('http://api.nautilus.com/api/v1/sprint3/websocket/stats');
  check(wsResponse, {
    'WebSocket stats available': (r) => r.status === 200,
  });

  sleep(1);
}
```

---

## Health Checks & Monitoring

### Health Check Script

```bash
#!/bin/bash
# scripts/health-check.sh

set -e

BASE_URL=${BASE_URL:-"http://localhost:8001"}
TIMEOUT=${TIMEOUT:-30}

echo "Running health checks for Sprint 3 deployment..."

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}
    
    echo "Checking $endpoint..."
    
    status_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time $TIMEOUT \
        "$BASE_URL$endpoint")
    
    if [ "$status_code" -eq "$expected_status" ]; then
        echo "✅ $endpoint: OK ($status_code)"
        return 0
    else
        echo "❌ $endpoint: FAILED ($status_code)"
        return 1
    fi
}

# Core health checks
check_endpoint "/health"
check_endpoint "/ready"

# Sprint 3 specific checks
check_endpoint "/api/v1/sprint3/system/health"
check_endpoint "/api/v1/sprint3/system/components"
check_endpoint "/api/v1/sprint3/websocket/stats"

# Database connectivity
check_endpoint "/api/v1/sprint3/system/database/health"

# Redis connectivity
check_endpoint "/api/v1/sprint3/system/redis/health"

# WebSocket connectivity test
echo "Testing WebSocket connectivity..."
if timeout 10s curl -s -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
    "$BASE_URL/ws/system/health" > /dev/null 2>&1; then
    echo "✅ WebSocket: OK"
else
    echo "❌ WebSocket: FAILED"
fi

echo "Health checks completed!"
```

### Deployment Verification Script

```bash
#!/bin/bash
# scripts/verify-deployment.sh

ENVIRONMENT=${1:-production}
NAMESPACE=${2:-nautilus-sprint3}

echo "Verifying Sprint 3 deployment in $ENVIRONMENT environment..."

# Check pod status
echo "Checking pod status..."
kubectl get pods -n $NAMESPACE

# Verify all pods are running
NOT_READY=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running -o name | wc -l)
if [ "$NOT_READY" -gt 0 ]; then
    echo "❌ Some pods are not ready:"
    kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
    exit 1
else
    echo "✅ All pods are running"
fi

# Check service connectivity
echo "Checking service connectivity..."
kubectl get services -n $NAMESPACE

# Verify ingress
echo "Checking ingress..."
kubectl get ingress -n $NAMESPACE

# Run application health checks
echo "Running application health checks..."
kubectl exec -n $NAMESPACE deployment/nautilus-backend -- \
    curl -f http://localhost:8001/health

echo "Deployment verification completed successfully!"
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Startup Issues

```bash
# Check container logs
docker-compose logs backend

# Check specific service logs
kubectl logs -n nautilus-sprint3 deployment/nautilus-backend

# Debug container startup
docker run -it --rm nautilus/backend:latest /bin/bash
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
docker exec -it nautilus-postgres psql -U nautilus -d nautilus -c "SELECT 1;"

# Check connection pool status
curl http://localhost:8001/api/v1/sprint3/system/database/pool-status
```

#### 3. WebSocket Connection Issues

```bash
# Check WebSocket service status
curl http://localhost:8001/api/v1/sprint3/websocket/stats

# Test WebSocket connection
wscat -c ws://localhost:8001/ws/system/health
```

#### 4. Performance Issues

```bash
# Check resource usage
docker stats

# Monitor application metrics
curl http://localhost:8001/metrics | grep -E "(cpu|memory|websocket)"

# Check database performance
kubectl exec -n nautilus-sprint3 postgres-0 -- \
    psql -U nautilus -d nautilus -c "SELECT * FROM pg_stat_activity;"
```

### Rollback Procedures

#### Docker Compose Rollback

```bash
# Rollback to previous version
docker-compose -f docker-compose.yml -f docker-compose.production.yml \
    down

# Deploy previous version
docker-compose -f docker-compose.yml -f docker-compose.production.yml \
    up -d --scale backend=3
```

#### Kubernetes Rollback

```bash
# Check deployment history
kubectl rollout history deployment/nautilus-backend -n nautilus-sprint3

# Rollback to previous version
kubectl rollout undo deployment/nautilus-backend -n nautilus-sprint3

# Rollback to specific revision
kubectl rollout undo deployment/nautilus-backend --to-revision=2 -n nautilus-sprint3
```

### Performance Optimization

#### Database Tuning

```sql
-- PostgreSQL configuration optimizations
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- TimescaleDB specific optimizations
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_retention_policy('market_data', INTERVAL '2 years');
```

#### Redis Optimization

```bash
# Redis configuration
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

This comprehensive deployment guide provides everything needed to successfully deploy and manage Sprint 3 infrastructure across all environments.
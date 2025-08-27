# Deployment Guide Template

**Status**: 📋 Template  
**Category**: Deployment Documentation  
**BMAD Package**: doc-automation v1.0.0

## Overview

**Brief deployment guide description in bold text explaining the deployment process, target environments, and prerequisites.**

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- **CPU**: 2+ cores (4+ cores recommended)
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Disk Space**: 50GB available space
- **Network**: Internet connection for downloading dependencies

### Required Software
- **Docker**: Version 20.10+ with Docker Compose
- **Git**: Version 2.20+ for source code management
- **Node.js**: Version 16+ (if applicable)
- **Python**: Version 3.8+ (if applicable)

### Access Requirements
- **Repository Access**: Git credentials for source code
- **Container Registry**: Access to Docker registry
- **Cloud Provider**: AWS/Azure/GCP account (for cloud deployment)
- **Domain/DNS**: Domain name and DNS management access

## Quick Start

### Local Development Setup
```bash
# Clone the repository
git clone https://github.com/your-org/your-project.git
cd your-project

# Copy environment configuration
cp .env.example .env

# Edit configuration values
nano .env

# Start services with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8080/health
```

### Production Deployment
```bash
# Set production environment
export ENVIRONMENT=production

# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify all services are healthy
docker-compose ps
curl https://your-domain.com/health
```

## Environment Configuration

### Environment Variables
```bash
# Application Settings
APP_NAME=your-application
APP_VERSION=1.0.0
ENVIRONMENT=production

# Database Configuration
DATABASE_HOST=db.your-domain.com
DATABASE_PORT=5432
DATABASE_NAME=your_app_db
DATABASE_USER=app_user
DATABASE_PASSWORD=secure_password_here

# Cache Configuration
REDIS_HOST=cache.your-domain.com
REDIS_PORT=6379
REDIS_PASSWORD=cache_password_here

# External Services
API_KEY_SERVICE_1=your_api_key_here
API_KEY_SERVICE_2=another_api_key_here

# Security Settings
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here
CORS_ORIGINS=https://your-frontend.com,https://admin.your-domain.com

# Monitoring
LOG_LEVEL=info
METRICS_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

### Configuration Files

#### docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - database
      - cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  database:
    image: postgres:13
    environment:
      POSTGRES_DB: ${DATABASE_NAME}
      POSTGRES_USER: ${DATABASE_USER}
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  cache:
    image: redis:6-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

#### docker-compose.prod.yml
```yaml
version: '3.8'

services:
  app:
    image: your-registry/your-app:${APP_VERSION}
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.app.rule=Host(`your-domain.com`)"
      - "traefik.http.services.app.loadbalancer.server.port=8080"
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  database:
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  cache:
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Deployment Methods

### Method 1: Docker Compose (Recommended)

#### Step 1: Prepare Environment
```bash
# Create project directory
sudo mkdir -p /opt/your-app
cd /opt/your-app

# Set proper permissions
sudo chown $USER:$USER /opt/your-app
```

#### Step 2: Deploy Application
```bash
# Download deployment files
curl -O https://raw.githubusercontent.com/your-org/your-project/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/your-org/your-project/main/docker-compose.prod.yml
curl -O https://raw.githubusercontent.com/your-org/your-project/main/.env.example

# Configure environment
cp .env.example .env
nano .env  # Edit configuration values

# Deploy services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### Step 3: Verify Deployment
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f app

# Test application
curl https://your-domain.com/health
```

### Method 2: Kubernetes Deployment

#### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://baltocdn.com/helm/signing.asc | sudo apt-key add -
sudo apt-get install apt-transport-https --yes
echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

#### Deployment Commands
```bash
# Add Helm repository (if using Helm charts)
helm repo add your-charts https://charts.your-domain.com
helm repo update

# Create namespace
kubectl create namespace your-app-prod

# Apply secrets
kubectl create secret generic app-secrets \
  --from-env-file=.env \
  --namespace=your-app-prod

# Deploy with Helm
helm install your-app your-charts/your-app \
  --namespace=your-app-prod \
  --values=production-values.yaml

# Verify deployment
kubectl get pods -n your-app-prod
kubectl get services -n your-app-prod
```

### Method 3: Cloud Provider Deployment

#### AWS ECS Deployment
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name your-app-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster your-app-cluster \
  --service-name your-app-service \
  --task-definition your-app:1 \
  --desired-count 2
```

## Database Setup

### PostgreSQL Setup
```bash
# Connect to database
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME}

# Create application schema
\i schema/create_tables.sql
\i schema/initial_data.sql
\i schema/create_indexes.sql

# Verify setup
\dt  # List tables
\q   # Quit
```

### Database Migration
```bash
# Run migrations (if using migration tools)
docker-compose exec app npm run migrate

# Or manually with SQL files
for migration in migrations/*.sql; do
  docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -f "$migration"
done
```

### Database Backup Setup
```bash
# Create backup script
cat << 'EOF' > /opt/your-app/backup-db.sh
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T database pg_dump -U ${DATABASE_USER} ${DATABASE_NAME} > /backups/db_backup_${BACKUP_DATE}.sql
# Keep only last 7 days of backups
find /backups -name "db_backup_*.sql" -mtime +7 -delete
EOF

chmod +x /opt/your-app/backup-db.sh

# Setup cron job for daily backups
echo "0 2 * * * /opt/your-app/backup-db.sh" | crontab -
```

## SSL/TLS Configuration

### Let's Encrypt with Certbot
```bash
# Install certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Generate certificates
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Setup automatic renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### Manual Certificate Setup
```bash
# Create SSL directory
sudo mkdir -p /opt/your-app/ssl

# Copy certificates
sudo cp your-domain.crt /opt/your-app/ssl/
sudo cp your-domain.key /opt/your-app/ssl/
sudo chmod 600 /opt/your-app/ssl/*.key
```

## Monitoring Setup

### Health Checks
```bash
# Create health check script
cat << 'EOF' > /opt/your-app/health-check.sh
#!/bin/bash
HEALTH_URL="https://your-domain.com/health"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "$(date): Application is healthy (HTTP $HTTP_STATUS)"
    exit 0
else
    echo "$(date): Application is unhealthy (HTTP $HTTP_STATUS)"
    # Send alert (email, Slack, etc.)
    exit 1
fi
EOF

chmod +x /opt/your-app/health-check.sh

# Run health check every 5 minutes
echo "*/5 * * * * /opt/your-app/health-check.sh >> /var/log/health-check.log 2>&1" | crontab -
```

### Log Aggregation
```bash
# Configure log rotation
sudo tee /etc/logrotate.d/your-app << EOF
/var/log/your-app/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 your-app your-app
}
EOF
```

### Prometheus Monitoring
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'your-app'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
    scrape_interval: 15s
```

## Security Hardening

### Firewall Configuration
```bash
# Install and configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow application specific ports (if needed)
sudo ufw allow 8080

# Enable firewall
sudo ufw --force enable
sudo ufw status verbose
```

### Container Security
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image your-registry/your-app:latest

# Run containers as non-root user
# Add to Dockerfile:
# USER 1000:1000
```

### Secret Management
```bash
# Use Docker secrets for sensitive data
echo "your_secret_password" | docker secret create db_password -
echo "your_api_key" | docker secret create api_key -

# Update compose file to use secrets
# secrets:
#   - db_password
#   - api_key
```

## Troubleshooting

### Common Issues

#### Issue: Container Won't Start
```bash
# Check container logs
docker-compose logs app

# Check resource usage
docker stats

# Verify environment variables
docker-compose exec app env | grep -E "(DATABASE|REDIS)"

# Test database connectivity
docker-compose exec app nc -zv database 5432
```

#### Issue: Application Not Accessible
```bash
# Check port binding
docker-compose ps
netstat -tulpn | grep :8080

# Check firewall rules
sudo ufw status

# Test local connectivity
curl -I http://localhost:8080/health

# Check DNS resolution
nslookup your-domain.com
```

#### Issue: Database Connection Errors
```bash
# Check database container
docker-compose logs database

# Test database connection
docker-compose exec database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} -c "SELECT version();"

# Check environment variables
docker-compose exec app env | grep DATABASE

# Verify network connectivity
docker-compose exec app nc -zv database 5432
```

#### Issue: High Memory Usage
```bash
# Monitor memory usage
docker stats --no-stream

# Check application memory leaks
docker-compose exec app top

# Review configuration
docker-compose exec app cat /proc/meminfo

# Adjust container limits
# Add to compose file:
# mem_limit: 512m
# memswap_limit: 512m
```

### Performance Optimization

#### Database Optimization
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Add indexes for frequent queries
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Update statistics
ANALYZE;
```

#### Application Optimization
```bash
# Enable production optimizations
export NODE_ENV=production
export PYTHON_ENV=production

# Configure connection pooling
# Database: max_connections=20
# Redis: max_connections=10

# Enable caching
# Application-level caching
# Database query caching
# CDN for static assets
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
```bash
# Check system health
./health-check.sh

# Monitor disk usage
df -h

# Check logs for errors
docker-compose logs --tail=100 | grep -i error
```

#### Weekly Tasks
```bash
# Update container images
docker-compose pull
docker-compose up -d

# Clean up unused containers/images
docker system prune -f

# Backup verification
./verify-backup.sh
```

#### Monthly Tasks
```bash
# Update operating system packages
sudo apt update && sudo apt upgrade -y

# Review security logs
sudo grep -i "authentication failure\|failed login" /var/log/auth.log

# Performance review
./generate-performance-report.sh

# Certificate renewal check
sudo certbot certificates
```

### Backup and Recovery

#### Backup Strategy
```bash
# Full backup script
cat << 'EOF' > /opt/your-app/full-backup.sh
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/full_backup_${BACKUP_DATE}"

mkdir -p "$BACKUP_DIR"

# Database backup
docker-compose exec -T database pg_dump -U ${DATABASE_USER} ${DATABASE_NAME} > "$BACKUP_DIR/database.sql"

# Application files backup
rsync -av /opt/your-app/ "$BACKUP_DIR/app/" --exclude=node_modules --exclude=.git

# Configuration backup
cp -r /etc/nginx/sites-available "$BACKUP_DIR/nginx/"
cp /etc/crontab "$BACKUP_DIR/crontab"

# Compress backup
tar -czf "${BACKUP_DIR}.tar.gz" -C /backups "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "Full backup completed: ${BACKUP_DIR}.tar.gz"
EOF

chmod +x /opt/your-app/full-backup.sh
```

#### Recovery Procedures
```bash
# Database recovery
docker-compose exec -T database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} < backup/database.sql

# Application files recovery
rsync -av backup/app/ /opt/your-app/

# Restart services
docker-compose restart

# Verify recovery
curl https://your-domain.com/health
```

## Rollback Procedures

### Application Rollback
```bash
# Tag current version before deployment
docker tag your-registry/your-app:latest your-registry/your-app:backup-$(date +%Y%m%d)

# Rollback to previous version
docker-compose stop app
docker-compose pull your-registry/your-app:v1.0.0  # previous version
docker-compose up -d app

# Verify rollback
curl https://your-domain.com/health
```

### Database Rollback
```bash
# Create rollback point
docker-compose exec -T database pg_dump -U ${DATABASE_USER} ${DATABASE_NAME} > rollback_point.sql

# Apply database rollback (if needed)
docker-compose exec -T database psql -U ${DATABASE_USER} -d ${DATABASE_NAME} < previous_version.sql
```

---

**Document Information**:
- **Version**: 1.0.0
- **Author**: [Your Name]
- **Environment**: Production Ready
- **Last Updated**: $(date)
- **Next Review**: [Date + 3 months]

**Generated by**: BMAD Documentation Template System  
**Template**: deployment-guide.md  

## Template Usage

This template should be customized by:
1. Replacing placeholder values with actual configuration
2. Adding project-specific deployment steps
3. Including actual environment variables and secrets
4. Updating troubleshooting guides with real issues
5. Adding monitoring and alerting specific to your stack

### BMAD Commands for Deployment Documentation

```bash
# Apply this template to new deployment guide
bmad apply template deployment-guide target=docs/deployment/new-service.md

# Validate deployment documentation standards
bmad run check-doc-health include_patterns="['docs/deployment/**']"

# Generate deployment cross-references
bmad run generate-doc-sitemap include_patterns="['docs/deployment/**']" group_by=category
```
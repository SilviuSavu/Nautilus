# Nautilus Trading Platform - Production Credentials Guide

⚠️ **SECURITY WARNING**: This template shows what credentials you need. NEVER store actual credentials in plain text files!

## 🔐 **CREDENTIAL COLLECTION CHECKLIST**

### **📊 Interactive Brokers (IB) - CRITICAL**
```bash
# Live Trading Account
IB_USERID=your_ib_username
IB_PASSWORD=your_ib_password
IB_TRADING_MODE=live  # or paper
IB_CLIENT_ID=1001     # Unique client ID
IB_GATEWAY_HOST=127.0.0.1
IB_GATEWAY_PORT=4001  # Live: 4001, Paper: 4002

# Optional: Secondary Account for Backup
IB_USERID_SECONDARY=backup_account
IB_PASSWORD_SECONDARY=backup_password
IB_CLIENT_ID_SECONDARY=1002
```

### **🏦 Database Configuration**
```bash
# Production PostgreSQL
PRODUCTION_DATABASE_URL=postgresql://user:password@prod-db:5432/nautilus_prod
PRODUCTION_DB_HOST=your-prod-db-host.com
PRODUCTION_DB_PORT=5432
PRODUCTION_DB_NAME=nautilus_production
PRODUCTION_DB_USER=nautilus_prod
PRODUCTION_DB_PASSWORD=strong_db_password

# Read Replica (Optional)
PRODUCTION_DB_REPLICA_HOST=replica-host.com
POSTGRES_REPLICA_PASSWORD=replica_password
```

### **🚀 Redis Configuration**
```bash
# Production Redis Cluster
PRODUCTION_REDIS_URL=redis://user:password@prod-redis:6379/0
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=strong_redis_password
REDIS_DATABASE=0
```

### **🔑 Security & Authentication**
```bash
# JWT Configuration
JWT_SECRET=your-256-bit-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# API Security
API_RATE_LIMIT=1000
CORS_ORIGINS=https://your-frontend-domain.com
ALLOWED_HOSTS=your-backend-domain.com

# Session Management
SESSION_SECRET=another-strong-secret-key
COOKIE_SECURE=true
COOKIE_HTTPONLY=true
```

### **📈 Exchange API Keys**
```bash
# Binance (Optional)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret

# Coinbase Pro (Optional)
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

# Alpha Vantage (Market Data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### **☁️ Cloud Provider Credentials**

#### **AWS Setup:**
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=nautilus-backups

# RDS Configuration
AWS_RDS_ENDPOINT=your-rds-endpoint.amazonaws.com
AWS_ELASTICACHE_ENDPOINT=your-elasticache.amazonaws.com
```

#### **Google Cloud Setup:**
```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_REGION=us-central1
GCP_SQL_INSTANCE=nautilus-production
```

#### **Azure Setup:**
```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=nautilus-production
AZURE_SQL_SERVER=nautilus-sql-server
AZURE_REDIS_HOST=nautilus-redis.redis.cache.windows.net
```

### **📊 Monitoring & Alerting**
```bash
# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_PASSWORD=strong_grafana_password

# Prometheus
PROMETHEUS_RETENTION=15d

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@your-domain.com
SMTP_PASSWORD=app_specific_password

# Slack Alerts (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#trading-alerts

# PagerDuty (Optional)
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key
```

### **💾 Backup & Storage**
```bash
# S3 Backup Configuration
BACKUP_S3_BUCKET=nautilus-backups
BACKUP_RETENTION_DAYS=90
BACKUP_ENCRYPTION_KEY=backup_encryption_key

# Database Backup
DB_BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
DB_BACKUP_RETENTION=30        # Keep 30 days
```

---

## 🛡️ **CREDENTIAL SECURITY BEST PRACTICES**

### **1. Secure Storage Methods:**
```bash
# Option 1: Cloud Provider Secret Manager
# AWS Secrets Manager, Google Secret Manager, Azure Key Vault

# Option 2: HashiCorp Vault
# Enterprise-grade secret management

# Option 3: Environment Variables (Production Containers)
# Never in config files, always in container environment

# Option 4: Encrypted Configuration Files
# PGP encrypted, decrypted at runtime
```

### **2. Access Control:**
- **Principle of Least Privilege**: Only necessary permissions
- **Role-Based Access**: Different keys for different services
- **Rotation Schedule**: Rotate credentials every 90 days
- **Audit Trail**: Log all credential access and changes

### **3. Emergency Procedures:**
```bash
# Credential Compromise Response Plan
1. Immediately revoke compromised credentials
2. Generate new credentials
3. Update all services with new credentials
4. Audit all systems for unauthorized access
5. Document incident and lessons learned
```

---

## 📋 **CREDENTIAL COLLECTION WORKFLOW**

### **Step 1: Interactive Brokers Setup**
1. **Open IB Account**: Apply for live trading account
2. **Fund Account**: Initial capital deposit
3. **Enable API Trading**: Configure API permissions
4. **Test Connection**: Verify connectivity with paper trading
5. **Document Credentials**: Securely store account details

### **Step 2: Cloud Infrastructure**
1. **Choose Provider**: AWS, GCP, or Azure
2. **Create Account**: Set up billing and projects
3. **Generate Keys**: Create service account credentials
4. **Set Permissions**: Configure IAM roles and policies
5. **Test Access**: Verify API access and permissions

### **Step 3: Third-Party Services**
1. **Market Data**: Subscribe to real-time data feeds
2. **Monitoring**: Set up external monitoring service
3. **Email Service**: Configure SMTP for alerts
4. **Backup Storage**: Set up cloud storage for backups
5. **DNS Service**: Configure domain and SSL certificates

### **Step 4: Security Hardening**
1. **Generate Secrets**: Create strong, unique passwords
2. **Configure 2FA**: Enable two-factor authentication
3. **Set IP Restrictions**: Limit access by IP address
4. **Create Backups**: Secure backup of all credentials
5. **Test Recovery**: Verify credential recovery procedures

---

## 🚨 **CREDENTIAL CHECKLIST BEFORE GO-LIVE**

### **Critical (Must Have):**
- [ ] IB account funded and API-enabled
- [ ] Production database credentials
- [ ] Cloud provider access configured
- [ ] JWT secrets generated
- [ ] SSL certificates obtained

### **Important (Should Have):**
- [ ] Redis cluster credentials
- [ ] Email/SMS alerting configured
- [ ] Backup storage credentials
- [ ] Monitoring service access
- [ ] DNS and domain configuration

### **Optional (Nice to Have):**
- [ ] Secondary broker accounts
- [ ] Additional exchange API keys
- [ ] Advanced monitoring services
- [ ] Professional email service
- [ ] CDN configuration

---

## 🎯 **NEXT ACTIONS**

1. **Review this template** and identify which credentials you need
2. **Start with IB account** setup (longest lead time)
3. **Choose cloud provider** and create accounts
4. **Generate security credentials** (JWT secrets, passwords)
5. **Set up secure credential storage** (never plain text!)
6. **Test each service** as you configure credentials
7. **Document everything** in your password manager

**Remember**: Security is paramount in financial systems. Invest time in proper credential management now to avoid disasters later!
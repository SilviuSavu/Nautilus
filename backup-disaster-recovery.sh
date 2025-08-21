#!/bin/bash
# Nautilus Trading Platform - Backup & Disaster Recovery Setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"; }
warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"; }

# Configuration
BACKUP_DIR="/var/backups/nautilus"
S3_BUCKET="${BACKUP_S3_BUCKET:-nautilus-backups}"
RETENTION_DAYS=30
DATABASE_NAME="${DATABASE_NAME:-nautilus_prod}"
DATABASE_USER="${DATABASE_USER:-nautilus}"

log "💾 Nautilus Trading Platform - Backup & Disaster Recovery Setup"
log "================================================================="

# Create backup directory structure
setup_backup_directories() {
    log "📁 Setting up backup directories..."
    
    mkdir -p "${BACKUP_DIR}/database"
    mkdir -p "${BACKUP_DIR}/config"
    mkdir -p "${BACKUP_DIR}/logs"
    mkdir -p "${BACKUP_DIR}/application"
    mkdir -p "/var/log/nautilus-backups"
    
    success "Backup directories created"
}

# Database backup functions
create_database_backup() {
    local backup_type="${1:-full}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${BACKUP_DIR}/database/${DATABASE_NAME}_${backup_type}_${timestamp}.sql"
    
    log "🗄️  Creating ${backup_type} database backup..."
    
    if [[ "$backup_type" == "full" ]]; then
        # Full database backup
        docker-compose exec -T postgres-primary pg_dump \
            -U "${DATABASE_USER}" \
            -h localhost \
            --clean \
            --create \
            --if-exists \
            "${DATABASE_NAME}" > "${backup_file}"
    else
        # Incremental backup using WAL files
        docker-compose exec -T postgres-primary pg_basebackup \
            -U "${DATABASE_USER}" \
            -D "${BACKUP_DIR}/database/incremental_${timestamp}" \
            -Ft \
            -z \
            -P
    fi
    
    # Compress backup
    gzip "${backup_file}" 2>/dev/null || true
    
    # Upload to S3 if configured
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        aws s3 cp "${backup_file}.gz" "s3://${S3_BUCKET}/database/" || warning "S3 upload failed"
    fi
    
    success "Database backup created: ${backup_file}.gz"
    echo "${backup_file}.gz"
}

# Application configuration backup
backup_application_config() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local config_backup="${BACKUP_DIR}/config/config_${timestamp}.tar.gz"
    
    log "⚙️  Backing up application configuration..."
    
    # Create configuration backup
    tar -czf "${config_backup}" \
        docker-compose.production.yml \
        .env.production \
        monitoring/ \
        nginx/ \
        scripts/ \
        2>/dev/null || warning "Some config files not found"
    
    # Upload to S3 if configured
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        aws s3 cp "${config_backup}" "s3://${S3_BUCKET}/config/" || warning "S3 upload failed"
    fi
    
    success "Configuration backup created: ${config_backup}"
    echo "${config_backup}"
}

# Application data backup
backup_application_data() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local data_backup="${BACKUP_DIR}/application/data_${timestamp}.tar.gz"
    
    log "📊 Backing up application data..."
    
    # Backup application volumes and persistent data
    docker run --rm \
        -v nautilus_market_data:/data/market_data \
        -v nautilus_trading_logs:/data/logs \
        -v nautilus_cache:/data/cache \
        -v "${BACKUP_DIR}/application:/backup" \
        busybox tar -czf "/backup/data_${timestamp}.tar.gz" /data
    
    # Upload to S3 if configured
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        aws s3 cp "${data_backup}" "s3://${S3_BUCKET}/application/" || warning "S3 upload failed"
    fi
    
    success "Application data backup created: ${data_backup}"
    echo "${data_backup}"
}

# Log backup
backup_logs() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_backup="${BACKUP_DIR}/logs/logs_${timestamp}.tar.gz"
    
    log "📋 Backing up application logs..."
    
    # Collect logs from containers
    mkdir -p "/tmp/nautilus_logs_${timestamp}"
    
    # Backend logs
    docker-compose logs --no-color nautilus-backend > "/tmp/nautilus_logs_${timestamp}/backend.log" 2>/dev/null || true
    
    # Frontend logs  
    docker-compose logs --no-color nautilus-frontend > "/tmp/nautilus_logs_${timestamp}/frontend.log" 2>/dev/null || true
    
    # Database logs
    docker-compose logs --no-color postgres-primary > "/tmp/nautilus_logs_${timestamp}/database.log" 2>/dev/null || true
    
    # IB Gateway logs
    docker-compose logs --no-color ib-gateway > "/tmp/nautilus_logs_${timestamp}/ib-gateway.log" 2>/dev/null || true
    
    # Create compressed archive
    tar -czf "${log_backup}" -C "/tmp" "nautilus_logs_${timestamp}"
    rm -rf "/tmp/nautilus_logs_${timestamp}"
    
    # Upload to S3 if configured
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        aws s3 cp "${log_backup}" "s3://${S3_BUCKET}/logs/" || warning "S3 upload failed"
    fi
    
    success "Log backup created: ${log_backup}"
    echo "${log_backup}"
}

# Full system backup
full_system_backup() {
    log "🔄 Performing full system backup..."
    
    local backup_start=$(date)
    local backup_manifest="${BACKUP_DIR}/backup_manifest_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Nautilus Trading Platform - Full System Backup" > "${backup_manifest}"
    echo "Backup Date: ${backup_start}" >> "${backup_manifest}"
    echo "=========================================" >> "${backup_manifest}"
    
    # Database backup
    local db_backup=$(create_database_backup "full")
    echo "Database Backup: ${db_backup}" >> "${backup_manifest}"
    
    # Configuration backup
    local config_backup=$(backup_application_config)
    echo "Configuration Backup: ${config_backup}" >> "${backup_manifest}"
    
    # Application data backup
    local data_backup=$(backup_application_data)
    echo "Application Data Backup: ${data_backup}" >> "${backup_manifest}"
    
    # Log backup
    local log_backup=$(backup_logs)
    echo "Log Backup: ${log_backup}" >> "${backup_manifest}"
    
    # System information
    echo "" >> "${backup_manifest}"
    echo "System Information:" >> "${backup_manifest}"
    echo "Hostname: $(hostname)" >> "${backup_manifest}"
    echo "Docker Version: $(docker --version)" >> "${backup_manifest}"
    echo "Compose Version: $(docker-compose --version)" >> "${backup_manifest}"
    
    # Docker images
    echo "" >> "${backup_manifest}"
    echo "Docker Images:" >> "${backup_manifest}"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" >> "${backup_manifest}"
    
    # Upload manifest to S3
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        aws s3 cp "${backup_manifest}" "s3://${S3_BUCKET}/" || warning "Manifest S3 upload failed"
    fi
    
    local backup_end=$(date)
    echo "Backup Completed: ${backup_end}" >> "${backup_manifest}"
    
    success "Full system backup completed successfully"
    success "Backup manifest: ${backup_manifest}"
}

# Cleanup old backups
cleanup_old_backups() {
    log "🧹 Cleaning up old backups (older than ${RETENTION_DAYS} days)..."
    
    # Local cleanup
    find "${BACKUP_DIR}" -type f -mtime +${RETENTION_DAYS} -name "*.gz" -delete
    find "${BACKUP_DIR}" -type f -mtime +${RETENTION_DAYS} -name "*.sql" -delete
    find "${BACKUP_DIR}" -type f -mtime +${RETENTION_DAYS} -name "*.txt" -delete
    
    # S3 cleanup (if configured)
    if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
        local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y-%m-%d)
        aws s3 ls "s3://${S3_BUCKET}/" --recursive | while read -r line; do
            file_date=$(echo "$line" | awk '{print $1}')
            file_path=$(echo "$line" | awk '{print $4}')
            
            if [[ "$file_date" < "$cutoff_date" ]]; then
                aws s3 rm "s3://${S3_BUCKET}/${file_path}" || warning "Failed to delete ${file_path}"
            fi
        done
    fi
    
    success "Old backup cleanup completed"
}

# Restore functions
restore_database() {
    local backup_file="${1}"
    
    if [[ ! -f "${backup_file}" ]]; then
        error "Backup file not found: ${backup_file}"
        return 1
    fi
    
    log "🔄 Restoring database from: ${backup_file}"
    
    # Stop application services
    docker-compose stop nautilus-backend nautilus-frontend
    
    # Restore database
    if [[ "${backup_file}" == *.gz ]]; then
        gunzip -c "${backup_file}" | docker-compose exec -T postgres-primary psql -U "${DATABASE_USER}"
    else
        docker-compose exec -T postgres-primary psql -U "${DATABASE_USER}" < "${backup_file}"
    fi
    
    # Restart services
    docker-compose start nautilus-backend nautilus-frontend
    
    success "Database restore completed"
}

# Disaster recovery test
test_disaster_recovery() {
    log "🚨 Testing disaster recovery procedures..."
    
    # Create test backup
    local test_backup=$(create_database_backup "test")
    
    # Simulate disaster by stopping services
    log "Simulating disaster scenario..."
    docker-compose stop
    
    # Wait a moment
    sleep 5
    
    # Restore services
    log "Restoring services from backup..."
    docker-compose up -d
    
    # Wait for services to start
    sleep 30
    
    # Test service health
    if curl -f http://localhost:8000/health &> /dev/null; then
        success "Disaster recovery test: PASSED"
    else
        error "Disaster recovery test: FAILED"
        return 1
    fi
    
    # Cleanup test backup
    rm -f "${test_backup}"
    
    success "Disaster recovery test completed successfully"
}

# Setup automated backup scheduling
setup_backup_schedule() {
    log "⏰ Setting up automated backup schedule..."
    
    # Create backup scripts directory
    mkdir -p "/opt/nautilus/scripts"
    
    # Create daily backup script
    cat > "/opt/nautilus/scripts/daily_backup.sh" << 'EOF'
#!/bin/bash
# Nautilus Trading Platform - Daily Backup Script

# Source this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../backup-disaster-recovery.sh"

# Log file
LOG_FILE="/var/log/nautilus-backups/daily_backup_$(date +%Y%m%d).log"

{
    echo "Starting daily backup at $(date)"
    full_system_backup
    cleanup_old_backups
    echo "Daily backup completed at $(date)"
} >> "${LOG_FILE}" 2>&1
EOF

    chmod +x "/opt/nautilus/scripts/daily_backup.sh"
    
    # Create weekly backup script
    cat > "/opt/nautilus/scripts/weekly_backup.sh" << 'EOF'
#!/bin/bash
# Nautilus Trading Platform - Weekly Backup Script

# Source this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../backup-disaster-recovery.sh"

# Log file
LOG_FILE="/var/log/nautilus-backups/weekly_backup_$(date +%Y%m%d).log"

{
    echo "Starting weekly backup at $(date)"
    full_system_backup
    test_disaster_recovery
    echo "Weekly backup completed at $(date)"
} >> "${LOG_FILE}" 2>&1
EOF

    chmod +x "/opt/nautilus/scripts/weekly_backup.sh"
    
    # Add cron jobs
    (crontab -l 2>/dev/null; echo "0 2 * * * /opt/nautilus/scripts/daily_backup.sh") | crontab -
    (crontab -l 2>/dev/null; echo "0 3 * * 0 /opt/nautilus/scripts/weekly_backup.sh") | crontab -
    
    success "Automated backup schedule configured"
    log "Daily backups: 2:00 AM every day"
    log "Weekly backups with DR test: 3:00 AM every Sunday"
}

# Create backup monitoring
setup_backup_monitoring() {
    log "📊 Setting up backup monitoring..."
    
    # Create backup monitoring script
    cat > "/opt/nautilus/scripts/backup_monitor.sh" << 'EOF'
#!/bin/bash
# Nautilus Trading Platform - Backup Monitoring

BACKUP_DIR="/var/backups/nautilus"
ALERT_EMAIL="${ALERT_EMAIL:-admin@nautilus-trading.com}"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"

check_recent_backup() {
    local backup_type="$1"
    local max_age_hours="$2"
    
    local latest_backup=$(find "${BACKUP_DIR}/${backup_type}" -name "*.gz" -mtime -1 | head -1)
    
    if [[ -z "${latest_backup}" ]]; then
        echo "ALERT: No recent ${backup_type} backup found (within ${max_age_hours} hours)"
        return 1
    fi
    
    echo "OK: Recent ${backup_type} backup found: ${latest_backup}"
    return 0
}

# Check all backup types
check_recent_backup "database" 24
check_recent_backup "config" 24  
check_recent_backup "application" 24
check_recent_backup "logs" 24

# Check S3 sync status if configured
if [[ -n "${AWS_ACCESS_KEY_ID}" ]] && command -v aws &> /dev/null; then
    if aws s3 ls "s3://${S3_BUCKET}/" &> /dev/null; then
        echo "OK: S3 backup sync operational"
    else
        echo "ALERT: S3 backup sync failed"
    fi
fi
EOF

    chmod +x "/opt/nautilus/scripts/backup_monitor.sh"
    
    # Add monitoring cron job
    (crontab -l 2>/dev/null; echo "0 6 * * * /opt/nautilus/scripts/backup_monitor.sh") | crontab -
    
    success "Backup monitoring configured"
}

# Main function
main() {
    case "${1:-setup}" in
        setup)
            setup_backup_directories
            setup_backup_schedule
            setup_backup_monitoring
            success "Backup & disaster recovery setup completed"
            ;;
        backup)
            full_system_backup
            ;;
        restore)
            if [[ -n "${2}" ]]; then
                restore_database "${2}"
            else
                error "Usage: $0 restore <backup_file>"
                exit 1
            fi
            ;;
        test)
            test_disaster_recovery
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            echo "Usage: $0 {setup|backup|restore|test|cleanup}"
            echo ""
            echo "Commands:"
            echo "  setup   - Initial backup system setup"
            echo "  backup  - Perform full system backup"
            echo "  restore - Restore from backup file"
            echo "  test    - Test disaster recovery procedures"
            echo "  cleanup - Clean up old backup files"
            exit 1
            ;;
    esac
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
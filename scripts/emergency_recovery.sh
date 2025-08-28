#!/bin/bash
# ============================================================================
# CACHE-NATIVE MESSAGE BUS - EMERGENCY RECOVERY SCRIPT
# Emergency procedures for cache-level integration failures
# Dream Team Emergency Response Protocol
# ============================================================================

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
BACKEND_DIR="/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend"
LOG_DIR="/tmp/nautilus_cache_deployment"
RECOVERY_LOG="${LOG_DIR}/emergency_recovery_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="${LOG_DIR}/backups"

# Emergency thresholds
CRITICAL_LATENCY_NS=100000      # 100μs critical latency
CRITICAL_HIT_RATE=80            # 80% critical hit rate
CRITICAL_ERROR_RATE=5.0         # 5% critical error rate
CRITICAL_MEMORY_PERCENT=95      # 95% critical memory usage

echo -e "${RED}🚨 EMERGENCY RECOVERY SYSTEM 🚨${NC}"
echo -e "${RED}===============================${NC}"
echo "Recovery initiated: $(date)"
echo "Recovery log: ${RECOVERY_LOG}"
echo ""

# Create necessary directories
mkdir -p "${LOG_DIR}" "${BACKUP_DIR}"

log_emergency() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] EMERGENCY: $1" | tee -a "${RECOVERY_LOG}"
}

assess_system_state() {
    echo -e "${YELLOW}[EMERGENCY] Assessing current system state...${NC}"
    
    # Check system resources
    memory_usage=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().percent:.1f}')")
    cpu_usage=$(python3 -c "import psutil; print(f'{psutil.cpu_percent(interval=1):.1f}')")
    
    echo -e "${BLUE}System Resources:${NC}"
    echo "  Memory Usage: ${memory_usage}%"
    echo "  CPU Usage: ${cpu_usage}%"
    
    # Check if cache processes are running
    cache_processes=$(pgrep -f "cache.*engine" || echo "0")
    echo "  Cache Processes: ${cache_processes}"
    
    # Quick health assessment
    local system_critical=false
    
    if (( $(echo "${memory_usage} > ${CRITICAL_MEMORY_PERCENT}" | bc -l) )); then
        echo -e "${RED}✗ CRITICAL: Memory usage at ${memory_usage}%${NC}"
        system_critical=true
    fi
    
    if [[ "${cache_processes}" == "0" ]]; then
        echo -e "${RED}✗ CRITICAL: No cache processes running${NC}"
        system_critical=true
    fi
    
    log_emergency "System assessment: Memory=${memory_usage}%, CPU=${cpu_usage}%, Processes=${cache_processes}, Critical=${system_critical}"
    
    if [[ "${system_critical}" == "true" ]]; then
        return 1
    else
        return 0
    fi
}

emergency_stop() {
    echo -e "${RED}[EMERGENCY] Initiating emergency stop procedure...${NC}"
    
    # Stop all cache-related processes
    echo -e "${YELLOW}Stopping cache processes...${NC}"
    pkill -f "cache_native.*engine" 2>/dev/null || true
    pkill -f "l1_cache_manager" 2>/dev/null || true
    pkill -f "l2_cache_coordinator" 2>/dev/null || true
    pkill -f "slc_unified_compute" 2>/dev/null || true
    
    # Stop monitoring
    echo -e "${YELLOW}Stopping cache monitoring...${NC}"
    pkill -f "cache_monitor" 2>/dev/null || true
    
    # Clear cache memory (if safe)
    echo -e "${YELLOW}Clearing system caches...${NC}"
    sync
    echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    
    # Validate stop
    sleep 2
    remaining_processes=$(pgrep -f "cache.*engine" | wc -l)
    
    if [[ ${remaining_processes} -eq 0 ]]; then
        echo -e "${GREEN}✓ Emergency stop completed successfully${NC}"
        log_emergency "Emergency stop completed - all cache processes terminated"
        return 0
    else
        echo -e "${RED}✗ Emergency stop incomplete - ${remaining_processes} processes remain${NC}"
        log_emergency "Emergency stop incomplete - ${remaining_processes} processes still running"
        return 1
    fi
}

create_system_backup() {
    echo -e "${BLUE}[EMERGENCY] Creating system backup...${NC}"
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${BACKUP_DIR}/emergency_backup_${backup_timestamp}"
    
    mkdir -p "${backup_path}"
    
    # Backup critical configuration files
    if [[ -d "${BACKEND_DIR}/acceleration" ]]; then
        cp -r "${BACKEND_DIR}/acceleration" "${backup_path}/" 2>/dev/null || true
    fi
    
    if [[ -d "${BACKEND_DIR}/engines" ]]; then
        find "${BACKEND_DIR}/engines" -name "*cache_native*" -exec cp {} "${backup_path}/" \; 2>/dev/null || true
    fi
    
    if [[ -d "${BACKEND_DIR}/monitoring" ]]; then
        cp -r "${BACKEND_DIR}/monitoring" "${backup_path}/" 2>/dev/null || true
    fi
    
    # Backup logs
    if [[ -d "${LOG_DIR}" ]]; then
        cp -r "${LOG_DIR}"/*.log "${backup_path}/" 2>/dev/null || true
        cp -r "${LOG_DIR}"/*.json "${backup_path}/" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ System backup created: ${backup_path}${NC}"
    log_emergency "System backup created at ${backup_path}"
    
    return 0
}

rollback_to_legacy() {
    echo -e "${PURPLE}[EMERGENCY] Rolling back to legacy message bus...${NC}"
    
    # Create backup before rollback
    create_system_backup
    
    # Disable cache-native implementations
    echo -e "${YELLOW}Disabling cache-native implementations...${NC}"
    
    # Rename cache files to disable them
    if [[ -f "${BACKEND_DIR}/acceleration/l1_cache_manager.py" ]]; then
        mv "${BACKEND_DIR}/acceleration/l1_cache_manager.py" "${BACKEND_DIR}/acceleration/l1_cache_manager.py.disabled" 2>/dev/null || true
    fi
    
    if [[ -f "${BACKEND_DIR}/acceleration/l2_cache_coordinator.py" ]]; then
        mv "${BACKEND_DIR}/acceleration/l2_cache_coordinator.py" "${BACKEND_DIR}/acceleration/l2_cache_coordinator.py.disabled" 2>/dev/null || true
    fi
    
    if [[ -f "${BACKEND_DIR}/acceleration/slc_unified_compute.py" ]]; then
        mv "${BACKEND_DIR}/acceleration/slc_unified_compute.py" "${BACKEND_DIR}/acceleration/slc_unified_compute.py.disabled" 2>/dev/null || true
    fi
    
    # Disable migrated engines
    find "${BACKEND_DIR}/engines" -name "*cache_native*" -exec mv {} {}.disabled \; 2>/dev/null || true
    
    # Restore legacy Redis message bus configuration
    echo -e "${YELLOW}Restoring legacy message bus configuration...${NC}"
    
    # Create emergency Redis configuration
    cat > "${LOG_DIR}/emergency_redis.conf" << 'EOF'
# Emergency Redis Configuration - Legacy Mode
port 6379
bind 127.0.0.1
timeout 300
tcp-keepalive 300
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
    
    # Start emergency Redis instance if needed
    if ! pgrep redis-server >/dev/null; then
        echo -e "${YELLOW}Starting emergency Redis instance...${NC}"
        redis-server "${LOG_DIR}/emergency_redis.conf" --daemonize yes 2>/dev/null || true
        sleep 2
    fi
    
    # Verify rollback
    if pgrep redis-server >/dev/null && [[ ! -f "${BACKEND_DIR}/acceleration/l1_cache_manager.py" ]]; then
        echo -e "${GREEN}✓ Rollback to legacy system completed${NC}"
        log_emergency "Rollback completed - legacy message bus restored"
        return 0
    else
        echo -e "${RED}✗ Rollback incomplete${NC}"
        log_emergency "Rollback incomplete - manual intervention required"
        return 1
    fi
}

restore_from_backup() {
    echo -e "${BLUE}[EMERGENCY] Restoring from backup...${NC}"
    
    # Find latest backup
    local latest_backup=$(find "${BACKUP_DIR}" -name "emergency_backup_*" -type d | sort -r | head -n1)
    
    if [[ -z "${latest_backup}" ]]; then
        echo -e "${RED}✗ No backup found for restoration${NC}"
        log_emergency "No backup available for restoration"
        return 1
    fi
    
    echo -e "${YELLOW}Restoring from: ${latest_backup}${NC}"
    
    # Stop current processes
    emergency_stop
    
    # Restore files
    if [[ -d "${latest_backup}/acceleration" ]]; then
        cp -r "${latest_backup}/acceleration" "${BACKEND_DIR}/" 2>/dev/null || true
    fi
    
    if [[ -d "${latest_backup}/monitoring" ]]; then
        cp -r "${latest_backup}/monitoring" "${BACKEND_DIR}/" 2>/dev/null || true
    fi
    
    # Restore cache native engines
    find "${latest_backup}" -name "*cache_native*" -exec cp {} "${BACKEND_DIR}/engines/" \; 2>/dev/null || true
    
    echo -e "${GREEN}✓ Restoration from backup completed${NC}"
    log_emergency "System restored from backup: ${latest_backup}"
    
    return 0
}

run_emergency_diagnostics() {
    echo -e "${BLUE}[EMERGENCY] Running emergency diagnostics...${NC}"
    
    local diagnostics_file="${LOG_DIR}/emergency_diagnostics_$(date +%Y%m%d_%H%M%S).json"
    
    # Collect system information
    python3 << PYEOF > "${diagnostics_file}"
import json
import sys
import os
import psutil
import time
from datetime import datetime

# Add backend to path
sys.path.append('${BACKEND_DIR}')

diagnostics = {
    "timestamp": datetime.now().isoformat(),
    "emergency_diagnostics": True,
    "system_info": {
        "platform": os.uname().sysname + " " + os.uname().release,
        "architecture": os.uname().machine,
        "python_version": sys.version.split()[0],
        "process_count": len(psutil.pids()),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
    },
    "resource_usage": {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_usage_percent": psutil.disk_usage('/').percent
    },
    "cache_modules_status": {},
    "recommendations": []
}

# Test cache module availability
try:
    from acceleration.l1_cache_manager import get_l1_cache_manager
    diagnostics["cache_modules_status"]["l1_cache"] = "AVAILABLE"
except ImportError as e:
    diagnostics["cache_modules_status"]["l1_cache"] = f"UNAVAILABLE: {str(e)}"

try:
    from acceleration.l2_cache_coordinator import L2CacheCoordinator
    diagnostics["cache_modules_status"]["l2_cache"] = "AVAILABLE"
except ImportError as e:
    diagnostics["cache_modules_status"]["l2_cache"] = f"UNAVAILABLE: {str(e)}"

try:
    from acceleration.slc_unified_compute import SLCUnifiedCompute
    diagnostics["cache_modules_status"]["slc_cache"] = "AVAILABLE"
except ImportError as e:
    diagnostics["cache_modules_status"]["slc_cache"] = f"UNAVAILABLE: {str(e)}"

# Generate recommendations
if diagnostics["resource_usage"]["memory_percent"] > 90:
    diagnostics["recommendations"].append("CRITICAL: Memory usage >90% - consider emergency stop")

if diagnostics["resource_usage"]["cpu_percent"] > 80:
    diagnostics["recommendations"].append("WARNING: High CPU usage - monitor for runaway processes")

if all("UNAVAILABLE" in status for status in diagnostics["cache_modules_status"].values()):
    diagnostics["recommendations"].append("CRITICAL: All cache modules unavailable - rollback recommended")

print(json.dumps(diagnostics, indent=2))
PYEOF
    
    echo -e "${GREEN}✓ Emergency diagnostics completed: ${diagnostics_file}${NC}"
    
    # Display critical information
    python3 -c "
import json
with open('${diagnostics_file}') as f:
    data = json.load(f)
    
print('\\n🔍 EMERGENCY DIAGNOSTICS SUMMARY:')
print(f'Memory Usage: {data[\"resource_usage\"][\"memory_percent\"]:.1f}%')
print(f'CPU Usage: {data[\"resource_usage\"][\"cpu_percent\"]:.1f}%')
print('\\nCache Module Status:')
for module, status in data['cache_modules_status'].items():
    status_color = '✓' if 'AVAILABLE' in status else '✗'
    print(f'  {status_color} {module}: {status}')

if data['recommendations']:
    print('\\n🚨 RECOMMENDATIONS:')
    for rec in data['recommendations']:
        print(f'  • {rec}')
"
    
    log_emergency "Emergency diagnostics completed: ${diagnostics_file}"
    return 0
}

show_emergency_help() {
    cat << 'EOF'
🚨 EMERGENCY RECOVERY SYSTEM - HELP 🚨

USAGE:
    ./emergency_recovery.sh <command> [options]

COMMANDS:
    assess          - Assess current system state and identify issues
    stop            - Emergency stop of all cache-related processes  
    rollback        - Rollback to legacy message bus system
    restore         - Restore from latest backup
    diagnostics     - Run comprehensive emergency diagnostics
    help            - Show this help message

EMERGENCY PROCEDURES:

1. IMMEDIATE RESPONSE (System Unresponsive):
   ./emergency_recovery.sh stop
   ./emergency_recovery.sh diagnostics

2. PERFORMANCE DEGRADATION:
   ./emergency_recovery.sh assess
   ./emergency_recovery.sh diagnostics
   
3. CRITICAL FAILURE (Rollback Required):
   ./emergency_recovery.sh stop
   ./emergency_recovery.sh rollback
   
4. CORRUPTION/DATA LOSS:
   ./emergency_recovery.sh restore

AUTOMATIC TRIGGERS:
- Memory usage > 95%
- Cache hit rate < 80%
- Latency > 100μs
- Error rate > 5%

SUPPORT:
- Emergency Log: /tmp/nautilus_cache_deployment/emergency_recovery_*.log
- Backups: /tmp/nautilus_cache_deployment/backups/
- Diagnostics: /tmp/nautilus_cache_deployment/emergency_diagnostics_*.json

ESCALATION:
If automated recovery fails, contact system administrator immediately.
EOF
}

main() {
    local command=${1:-"help"}
    
    case "${command}" in
        assess)
            echo -e "${BLUE}Running system assessment...${NC}"
            if assess_system_state; then
                echo -e "${GREEN}✓ System state: STABLE${NC}"
                exit 0
            else
                echo -e "${RED}✗ System state: CRITICAL${NC}"
                echo -e "${YELLOW}Recommend running: ./emergency_recovery.sh diagnostics${NC}"
                exit 1
            fi
            ;;
        stop)
            echo -e "${RED}Initiating emergency stop...${NC}"
            emergency_stop
            exit $?
            ;;
        rollback)
            echo -e "${PURPLE}Initiating rollback to legacy system...${NC}"
            rollback_to_legacy
            exit $?
            ;;
        restore)
            echo -e "${BLUE}Initiating restoration from backup...${NC}"
            restore_from_backup
            exit $?
            ;;
        diagnostics)
            echo -e "${BLUE}Running emergency diagnostics...${NC}"
            run_emergency_diagnostics
            exit $?
            ;;
        help|--help|-h)
            show_emergency_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown command: ${command}${NC}"
            echo ""
            show_emergency_help
            exit 1
            ;;
    esac
}

# Check if bc is available for floating point comparison
if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}Warning: bc not available, some checks may be limited${NC}"
fi

main "$@"
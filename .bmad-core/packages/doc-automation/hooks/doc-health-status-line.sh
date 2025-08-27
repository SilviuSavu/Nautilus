#!/bin/bash
# BMAD Doc Health Status Line Integration
# Real-time documentation health display in Claude Code status bar

set -e

# Configuration
BMAD_ENABLED=${BMAD_ENABLED:-true}
STATUS_CACHE="/tmp/bmad-doc-health-cache"
BMAD_AGENT="doc-health"
UPDATE_INTERVAL=${DOC_HEALTH_UPDATE_INTERVAL:-300}  # 5 minutes default

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Health status icons
EXCELLENT_ICON="🎉"
GOOD_ICON="✅"
WARNING_ICON="⚠️"
CRITICAL_ICON="❌"
UNKNOWN_ICON="❓"

# Logging function
bmad_log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$BMAD_ENABLED" = true ]; then
        echo "[$timestamp] [$BMAD_AGENT] [$level] $message" >> /tmp/bmad-status-line.log
    fi
}

# Get current documentation health score
get_doc_health() {
    if [ ! -f "$STATUS_CACHE" ]; then
        echo "0:❓:unknown"
        return 1
    fi
    
    local cache_content=$(cat "$STATUS_CACHE" 2>/dev/null || echo "")
    if [ -z "$cache_content" ]; then
        echo "0:❓:unknown"
        return 1
    fi
    
    # Parse cache format: timestamp:score%:status or timestamp:score%:broken_links
    local timestamp=$(echo "$cache_content" | cut -d':' -f1)
    local score_data=$(echo "$cache_content" | cut -d':' -f2-)
    
    # Check if cache is stale (older than update interval)
    local current_time=$(date +%s)
    local cache_age=$((current_time - timestamp))
    
    if [ $cache_age -gt $UPDATE_INTERVAL ]; then
        # Cache is stale, trigger update in background
        nohup "$0" --update-cache > /dev/null 2>&1 &
    fi
    
    echo "$score_data"
}

# Update documentation health cache
update_doc_health_cache() {
    bmad_log "INFO" "Updating documentation health cache"
    
    # Create a lock file to prevent concurrent updates
    local lock_file="/tmp/bmad-doc-health-update.lock"
    if [ -f "$lock_file" ]; then
        local lock_age=$(( $(date +%s) - $(stat -f "%m" "$lock_file" 2>/dev/null || stat -c "%Y" "$lock_file" 2>/dev/null || echo "0") ))
        if [ $lock_age -lt 60 ]; then
            bmad_log "DEBUG" "Update already in progress (lock age: ${lock_age}s)"
            return 0
        else
            rm -f "$lock_file"
        fi
    fi
    
    echo $$ > "$lock_file"
    
    # Run quick health check
    local project_root=$(pwd)
    local total_files=0
    local large_files=0
    local oversized_files=0
    local critical_files=0
    
    # Quick file size analysis
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            local size=$(wc -c < "$file" 2>/dev/null || echo "0")
            ((total_files++))
            
            if [ $size -gt 20000 ]; then
                ((critical_files++))
            elif [ $size -gt 15000 ]; then
                ((oversized_files++))
            elif [ $size -gt 10000 ]; then
                ((large_files++))
            fi
        fi
    done < <(find . -name "*.md" -type f \
        -not -path "./docs/archive/*" \
        -not -path "./node_modules/*" \
        -not -path "./.git/*" \
        -not -path "./.bmad-core/packages/*/docs/*" 2>/dev/null | head -100)
    
    # Calculate health score
    local health_score=100
    if [ $total_files -gt 0 ]; then
        local penalty=$((critical_files * 15 + oversized_files * 8 + large_files * 3))
        health_score=$(( 100 - (penalty * 100 / total_files) ))
        if [ $health_score -lt 0 ]; then
            health_score=0
        fi
    fi
    
    # Determine status
    local status="unknown"
    local icon="$UNKNOWN_ICON"
    
    if [ $health_score -ge 95 ]; then
        status="excellent"
        icon="$EXCELLENT_ICON"
    elif [ $health_score -ge 85 ]; then
        status="good"
        icon="$GOOD_ICON"
    elif [ $health_score -ge 70 ]; then
        status="warning"
        icon="$WARNING_ICON"
    else
        status="critical"
        icon="$CRITICAL_ICON"
    fi
    
    # Update cache
    echo "$(date +%s):${health_score}%:${icon}:${status}" > "$STATUS_CACHE"
    
    bmad_log "INFO" "Health cache updated: ${health_score}% ($status)"
    
    # Cleanup
    rm -f "$lock_file"
}

# Generate status line text
generate_status_line() {
    local health_data=$(get_doc_health)
    local score=$(echo "$health_data" | cut -d':' -f1 | sed 's/%//')
    local icon=$(echo "$health_data" | cut -d':' -f2)
    local status=$(echo "$health_data" | cut -d':' -f3)
    
    if [ -z "$score" ] || [ "$score" = "unknown" ]; then
        echo "📚 Docs: $UNKNOWN_ICON Unknown"
        return
    fi
    
    # Format based on score
    local color_code=""
    case "$status" in
        "excellent") color_code="32" ;;  # Green
        "good") color_code="32" ;;       # Green
        "warning") color_code="33" ;;    # Yellow
        "critical") color_code="31" ;;   # Red
        *) color_code="37" ;;           # White
    esac
    
    # Generate different formats for different contexts
    case "${1:-default}" in
        "short")
            echo "$icon $score"
            ;;
        "detailed")
            echo "📚 Docs: $icon $score ($status)"
            ;;
        "colored")
            echo -e "\033[${color_code}m📚 Docs: $icon $score\033[0m"
            ;;
        "json")
            echo "{\"component\":\"docs\",\"score\":$score,\"status\":\"$status\",\"icon\":\"$icon\"}"
            ;;
        *)
            echo "📚 $icon $score"
            ;;
    esac
}

# Handle file change events
handle_file_change() {
    local changed_file="$1"
    local event_type="$2"
    
    bmad_log "DEBUG" "File change detected: $changed_file ($event_type)"
    
    # Only react to markdown file changes
    if [[ "$changed_file" =~ \.md$ ]]; then
        # Delay update to avoid rapid consecutive updates
        sleep 2
        
        # Check if another update is already scheduled
        if [ ! -f "/tmp/bmad-doc-update-pending" ]; then
            touch "/tmp/bmad-doc-update-pending"
            
            # Update cache in background after delay
            (
                sleep 5
                update_doc_health_cache
                rm -f "/tmp/bmad-doc-update-pending"
            ) &
            
            bmad_log "INFO" "Scheduled doc health update due to file change: $changed_file"
        fi
    fi
}

# Install file system watcher (if available)
install_watcher() {
    if command -v fswatch > /dev/null 2>&1; then
        bmad_log "INFO" "Installing fswatch-based file watcher"
        
        # Watch for markdown file changes
        fswatch -o --event Created --event Updated --event Removed \
            --exclude="/.git/" \
            --exclude="/node_modules/" \
            --exclude="/.bmad-core/packages/.*/docs/" \
            --include="\.md$" \
            . | while read num; do
                handle_file_change "unknown" "fswatch"
            done &
        
        echo $! > /tmp/bmad-doc-watcher.pid
        bmad_log "INFO" "File watcher started (PID: $(cat /tmp/bmad-doc-watcher.pid))"
        
    elif command -v inotifywait > /dev/null 2>&1; then
        bmad_log "INFO" "Installing inotify-based file watcher"
        
        # Watch for markdown file changes with inotify
        inotifywait -m -r -e create,modify,delete --format '%w%f %e' \
            --exclude '\.(git|node_modules)' \
            . | while read file event; do
                if [[ "$file" =~ \.md$ ]]; then
                    handle_file_change "$file" "$event"
                fi
            done &
        
        echo $! > /tmp/bmad-doc-watcher.pid
        bmad_log "INFO" "File watcher started (PID: $(cat /tmp/bmad-doc-watcher.pid))"
        
    else
        bmad_log "WARN" "No file system watcher available (fswatch or inotify-tools needed)"
        bmad_log "INFO" "Falling back to periodic updates every $UPDATE_INTERVAL seconds"
        
        # Fallback to periodic updates
        while true; do
            sleep $UPDATE_INTERVAL
            update_doc_health_cache
        done &
        
        echo $! > /tmp/bmad-doc-watcher.pid
    fi
}

# Stop file system watcher
stop_watcher() {
    if [ -f "/tmp/bmad-doc-watcher.pid" ]; then
        local watcher_pid=$(cat /tmp/bmad-doc-watcher.pid)
        if kill -0 "$watcher_pid" > /dev/null 2>&1; then
            kill "$watcher_pid"
            bmad_log "INFO" "Stopped file watcher (PID: $watcher_pid)"
        fi
        rm -f /tmp/bmad-doc-watcher.pid
    fi
}

# Handle script arguments
case "${1:-help}" in
    "--status"|"-s")
        generate_status_line "${2:-default}"
        ;;
    "--status-short")
        generate_status_line "short"
        ;;
    "--status-detailed")
        generate_status_line "detailed"
        ;;
    "--status-colored")
        generate_status_line "colored"
        ;;
    "--status-json")
        generate_status_line "json"
        ;;
    "--update-cache"|"-u")
        update_doc_health_cache
        ;;
    "--install-watcher"|"-w")
        if [ "$BMAD_ENABLED" = true ]; then
            install_watcher
            echo "File watcher installed. Documentation health will update automatically."
        else
            echo "BMAD is disabled. Enable with: export BMAD_ENABLED=true"
        fi
        ;;
    "--stop-watcher")
        stop_watcher
        echo "File watcher stopped."
        ;;
    "--test")
        echo -e "${BLUE}Testing BMAD Doc Health Status Line${NC}"
        echo "=================================="
        echo ""
        echo "Current status:"
        generate_status_line "colored"
        echo ""
        echo "All formats:"
        echo "  Short: $(generate_status_line "short")"
        echo "  Default: $(generate_status_line "default")"
        echo "  Detailed: $(generate_status_line "detailed")"
        echo "  JSON: $(generate_status_line "json")"
        echo ""
        echo "Cache file: $STATUS_CACHE"
        if [ -f "$STATUS_CACHE" ]; then
            echo "Cache content: $(cat "$STATUS_CACHE")"
            local cache_age=$(( $(date +%s) - $(cat "$STATUS_CACHE" | cut -d':' -f1) ))
            echo "Cache age: ${cache_age}s"
        else
            echo "Cache: Not found"
        fi
        ;;
    "--daemon"|"-d")
        echo -e "${BLUE}Starting BMAD Doc Health Status Line Daemon${NC}"
        bmad_log "INFO" "Starting status line daemon"
        
        # Initial cache update
        update_doc_health_cache
        
        # Install file watcher
        if [ "$BMAD_ENABLED" = true ]; then
            install_watcher
        fi
        
        # Keep daemon running
        echo "Status line daemon running. Press Ctrl+C to stop."
        trap "stop_watcher; exit 0" INT TERM
        
        while true; do
            sleep 60
            # Periodic health check to ensure cache is fresh
            if [ ! -f "$STATUS_CACHE" ]; then
                update_doc_health_cache
            fi
        done
        ;;
    "--help"|"-h"|*)
        echo -e "${BLUE}BMAD Doc Health Status Line Integration${NC}"
        echo "========================================"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  -s, --status [FORMAT]     Show current doc health status"
        echo "      --status-short        Show short format status"
        echo "      --status-detailed     Show detailed format status" 
        echo "      --status-colored      Show colored format status"
        echo "      --status-json         Show JSON format status"
        echo "  -u, --update-cache        Force update health cache"
        echo "  -w, --install-watcher     Install file system watcher"
        echo "      --stop-watcher        Stop file system watcher"
        echo "  -d, --daemon              Run as daemon with file watching"
        echo "      --test                Test all status formats"
        echo "  -h, --help                Show this help message"
        echo ""
        echo "Status Formats:"
        echo "  short     : '🎉 95%'"
        echo "  default   : '📚 🎉 95%'"
        echo "  detailed  : '📚 Docs: 🎉 95% (excellent)'"
        echo "  colored   : Colored terminal output"
        echo "  json      : {\"component\":\"docs\",\"score\":95,...}"
        echo ""
        echo "Environment Variables:"
        echo "  BMAD_ENABLED                 Enable BMAD integration (default: true)"
        echo "  DOC_HEALTH_UPDATE_INTERVAL   Update interval in seconds (default: 300)"
        echo ""
        echo "Claude Code Integration:"
        echo "  Add to your Claude Code status line configuration:"
        echo "  \"$0 --status-short\" for compact display"
        echo "  \"$0 --status\" for standard display"
        echo ""
        echo "Examples:"
        echo "  $0 --status                  # Show current health"
        echo "  $0 --daemon                  # Run with auto-updates"
        echo "  $0 --install-watcher         # Install file watcher only"
        echo "  watch -n 5 '$0 --status-colored'  # Monitor with color"
        ;;
esac

# If no arguments provided and running interactively, show status
if [ $# -eq 0 ] && [ -t 1 ]; then
    generate_status_line "colored"
fi
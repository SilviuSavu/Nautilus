#!/bin/bash
# BMAD Post-Edit Hook
# Automatically triggered after file edits to maintain documentation health

set -e

# Configuration
BMAD_ENABLED=${BMAD_ENABLED:-true}
BMAD_AGENT="doc-health"
HOOK_LOG="/tmp/bmad-post-edit.log"
AUTO_FIX_ENABLED=${BMAD_AUTO_FIX:-false}
VALIDATE_ON_EDIT=${BMAD_VALIDATE_ON_EDIT:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging function
bmad_log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$BMAD_AGENT] [$level] $message" >> "$HOOK_LOG"
    
    # Also output to stderr for debugging if enabled
    if [ "${BMAD_DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[$timestamp] [$BMAD_AGENT] [$level]${NC} $message" >&2
    fi
}

# BMAD notification function
bmad_notify() {
    local event="$1"
    local data="$2"
    
    if [ "$BMAD_ENABLED" = true ]; then
        echo "BMAD_EVENT:$event:$data:$(date +%s)" >> /tmp/bmad-hook-events.log
    fi
}

# Check if file is documentation
is_documentation_file() {
    local file="$1"
    
    [[ "$file" =~ \.(md|markdown)$ ]] && \
    [[ ! "$file" =~ /\.git/ ]] && \
    [[ ! "$file" =~ /node_modules/ ]] && \
    [[ ! "$file" =~ /\.bmad-core/packages/.*/docs/ ]]
}

# Get file health status
get_file_health() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        echo "missing"
        return 1
    fi
    
    local size=$(wc -c < "$file")
    local health="excellent"
    
    # Check file size
    if [ $size -gt 20000 ]; then
        health="critical"
    elif [ $size -gt 15000 ]; then
        health="oversized"
    elif [ $size -gt 10000 ]; then
        health="large"
    elif [ $size -lt 100 ]; then
        health="minimal"
    fi
    
    # Check for missing H1 heading
    if [ "$health" != "critical" ] && ! grep -q "^# " "$file" 2>/dev/null; then
        health="no_heading"
    fi
    
    # Check for multiple H1 headings
    if [ "$health" != "critical" ]; then
        local h1_count=$(grep -c "^# " "$file" 2>/dev/null || echo "0")
        if [ "$h1_count" -gt 1 ]; then
            health="multiple_h1"
        fi
    fi
    
    echo "$health"
}

# Quick link validation for changed file
validate_file_links() {
    local file="$1"
    local broken_links=0
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    bmad_log "DEBUG" "Validating links in $file"
    
    # Extract markdown links and validate internal ones
    while IFS= read -r line; do
        echo "$line" | grep -oE '\[[^]]*\]\([^)]+\)' | while IFS= read -r link; do
            local url="$(echo "$link" | sed -n 's/.*](\([^)]*\)).*/\1/p')"
            
            if [ -z "$url" ] || [[ "$url" =~ ^https?:// ]] || [[ "$url" =~ ^mailto: ]]; then
                continue  # Skip external and mailto links for speed
            fi
            
            # Check internal file links
            local target_file="$url"
            if [[ "$url" =~ # ]]; then
                target_file="${url%#*}"
            fi
            
            if [ -z "$target_file" ]; then
                target_file="$file"
            fi
            
            # Handle relative paths
            if [[ ! "$target_file" =~ ^/ ]]; then
                local source_dir="$(dirname "$file")"
                target_file="$source_dir/$target_file"
            fi
            
            # Normalize path
            target_file="$(cd "$(dirname "$target_file")" 2>/dev/null && pwd)/$(basename "$target_file")" 2>/dev/null || echo "$target_file"
            
            if [ ! -f "$target_file" ] && [ ! -d "$target_file" ]; then
                ((broken_links++))
                bmad_log "WARN" "Broken link in $file: $url"
            fi
        done
    done < "$file"
    
    return $broken_links
}

# Auto-fix common issues
auto_fix_file() {
    local file="$1"
    local fixes_applied=0
    
    if [ ! -f "$file" ] || [ "$AUTO_FIX_ENABLED" != "true" ]; then
        return 0
    fi
    
    bmad_log "INFO" "Attempting auto-fixes for $file"
    
    # Create backup
    cp "$file" "${file}.bmad-backup"
    
    # Fix 1: Add missing H1 heading if file has content but no H1
    if ! grep -q "^# " "$file" && [ "$(wc -l < "$file")" -gt 3 ]; then
        local filename=$(basename "$file" .md)
        local title=$(echo "$filename" | sed 's/-/ /g' | sed 's/\b\w/\U&/g')
        
        # Insert H1 at the beginning
        sed -i.tmp "1i\\
# $title\\
" "$file"
        rm -f "${file}.tmp"
        
        ((fixes_applied++))
        bmad_log "INFO" "Added H1 heading to $file"
    fi
    
    # Fix 2: Fix multiple H1 headings (convert subsequent ones to H2)
    local h1_count=$(grep -c "^# " "$file" 2>/dev/null || echo "0")
    if [ "$h1_count" -gt 1 ]; then
        # Convert all H1 except the first to H2
        awk '/^# / && !first_h1 { first_h1=1; print; next } /^# / { sub(/^# /, "## "); print; next } { print }' "$file" > "${file}.tmp"
        mv "${file}.tmp" "$file"
        
        ((fixes_applied++))
        bmad_log "INFO" "Fixed multiple H1 headings in $file"
    fi
    
    # Fix 3: Ensure consistent line endings
    if command -v dos2unix > /dev/null 2>&1; then
        dos2unix "$file" > /dev/null 2>&1
    fi
    
    # Fix 4: Remove trailing whitespace
    sed -i.tmp 's/[[:space:]]*$//' "$file"
    rm -f "${file}.tmp"
    
    if [ $fixes_applied -gt 0 ]; then
        bmad_log "INFO" "Applied $fixes_applied auto-fixes to $file"
        bmad_notify "AUTO_FIX" "$file:$fixes_applied"
    else
        # Remove backup if no changes made
        rm -f "${file}.bmad-backup"
    fi
    
    return $fixes_applied
}

# Update documentation health cache
update_health_cache() {
    bmad_log "DEBUG" "Updating documentation health cache"
    
    # Use the status line script to update cache
    local status_script="$(dirname "$0")/doc-health-status-line.sh"
    if [ -f "$status_script" ]; then
        "$status_script" --update-cache
    fi
}

# Process file change event
process_file_change() {
    local file="$1"
    local event_type="${2:-edit}"
    
    if ! is_documentation_file "$file"; then
        bmad_log "DEBUG" "Ignoring non-documentation file: $file"
        return 0
    fi
    
    bmad_log "INFO" "Processing file change: $file ($event_type)"
    
    # Get file health before any fixes
    local initial_health=$(get_file_health "$file")
    bmad_log "INFO" "File health: $file -> $initial_health"
    
    # Apply auto-fixes if enabled
    local fixes_applied=0
    if [ "$AUTO_FIX_ENABLED" = "true" ]; then
        auto_fix_file "$file"
        fixes_applied=$?
    fi
    
    # Validate links if enabled
    local broken_links=0
    if [ "$VALIDATE_ON_EDIT" = "true" ] && [ -f "$file" ]; then
        validate_file_links "$file"
        broken_links=$?
    fi
    
    # Get final health status
    local final_health=$(get_file_health "$file")
    
    # Log results
    if [ $fixes_applied -gt 0 ]; then
        bmad_log "INFO" "Auto-fixes applied: $fixes_applied ($initial_health -> $final_health)"
    fi
    
    if [ $broken_links -gt 0 ]; then
        bmad_log "WARN" "Broken links found: $broken_links in $file"
    fi
    
    # Notify BMAD system
    bmad_notify "FILE_PROCESSED" "$file:$event_type:$initial_health:$final_health:$fixes_applied:$broken_links"
    
    # Schedule health cache update (debounced)
    if [ ! -f "/tmp/bmad-cache-update-pending" ]; then
        touch "/tmp/bmad-cache-update-pending"
        (
            sleep 3  # Debounce multiple rapid changes
            update_health_cache
            rm -f "/tmp/bmad-cache-update-pending"
        ) &
    fi
    
    # Output summary if running interactively
    if [ -t 1 ] && [ "${BMAD_QUIET:-false}" != "true" ]; then
        local status_color=""
        case "$final_health" in
            "excellent"|"good") status_color="$GREEN" ;;
            "large"|"minimal"|"no_heading") status_color="$YELLOW" ;;
            "critical"|"oversized"|"multiple_h1") status_color="$RED" ;;
            *) status_color="$NC" ;;
        esac
        
        echo -e "${BLUE}BMAD Doc Health:${NC} ${status_color}$file${NC} ($final_health)"
        
        if [ $fixes_applied -gt 0 ]; then
            echo -e "${GREEN}  ✅ Applied $fixes_applied auto-fixes${NC}"
        fi
        
        if [ $broken_links -gt 0 ]; then
            echo -e "${YELLOW}  ⚠️ Found $broken_links broken links${NC}"
        fi
    fi
}

# Main execution
main() {
    if [ "$BMAD_ENABLED" != "true" ]; then
        bmad_log "DEBUG" "BMAD is disabled, skipping post-edit hook"
        return 0
    fi
    
    local file="${1:-}"
    local event_type="${2:-edit}"
    
    if [ -z "$file" ]; then
        echo "Usage: $0 <file> [event_type]"
        echo ""
        echo "Event types:"
        echo "  edit     - File was edited (default)"
        echo "  create   - File was created"
        echo "  delete   - File was deleted"
        echo "  rename   - File was renamed"
        echo ""
        echo "Environment Variables:"
        echo "  BMAD_ENABLED         Enable BMAD integration (default: true)"
        echo "  BMAD_AUTO_FIX        Enable automatic fixes (default: false)"
        echo "  BMAD_VALIDATE_ON_EDIT Validate links on edit (default: true)"
        echo "  BMAD_DEBUG           Enable debug output (default: false)"
        echo "  BMAD_QUIET           Suppress interactive output (default: false)"
        return 1
    fi
    
    bmad_log "INFO" "Post-edit hook triggered: $file ($event_type)"
    
    case "$event_type" in
        "delete")
            bmad_log "INFO" "File deleted: $file"
            bmad_notify "FILE_DELETED" "$file"
            update_health_cache
            ;;
        "create"|"edit"|"rename"|*)
            process_file_change "$file" "$event_type"
            ;;
    esac
}

# Handle script execution
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script is being executed directly
    main "$@"
else
    # Script is being sourced
    bmad_log "DEBUG" "Post-edit hook loaded as library"
fi
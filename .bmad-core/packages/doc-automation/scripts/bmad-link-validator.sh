#!/bin/bash
# BMAD Link Validation System
# Enhanced link validation with BMAD integration and intelligent reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REPORT_FILE="docs/automation/bmad-link-validation-report.md"
BMAD_LOG_FILE="docs/automation/bmad-link-validation.log"
CACHE_FILE="/tmp/bmad-link-cache"
MAX_PARALLEL=10  # Maximum parallel link checks

# BMAD Integration
BMAD_ENABLED=${BMAD_ENABLED:-true}
BMAD_AGENT="doc-health"
BMAD_PROJECT_PATH=${BMAD_PROJECT_PATH:-$(pwd)}

# Link validation settings
VALIDATE_EXTERNAL=${VALIDATE_EXTERNAL:-true}
VALIDATE_INTERNAL=${VALIDATE_INTERNAL:-true}
VALIDATE_ANCHORS=${VALIDATE_ANCHORS:-true}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-10}
USER_AGENT="BMAD-Link-Validator/1.0.0"

# Logging function for BMAD integration
bmad_log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$BMAD_ENABLED" = true ]; then
        echo "[$timestamp] [$BMAD_AGENT] [$level] $message" >> "$BMAD_LOG_FILE"
    fi
}

# BMAD notification function
bmad_notify() {
    local event="$1"
    local data="$2"
    
    if [ "$BMAD_ENABLED" = true ]; then
        echo "BMAD_EVENT:$event:$data" >> /tmp/bmad-link-events.log
    fi
}

# Validate external URL
validate_external_url() {
    local url="$1"
    local source_file="$2"
    
    # Check cache first
    local cache_key="$(echo "$url" | md5sum | cut -d' ' -f1)"
    if [ -f "$CACHE_FILE" ] && grep -q "^$cache_key:200:" "$CACHE_FILE" 2>/dev/null; then
        local cached_age=$(( $(date +%s) - $(grep "^$cache_key:200:" "$CACHE_FILE" | cut -d':' -f3) ))
        if [ $cached_age -lt 3600 ]; then  # Cache valid for 1 hour
            return 0
        fi
    fi
    
    # Validate URL with timeout and proper headers
    local status_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SECONDS" \
        --user-agent "$USER_AGENT" \
        --location \
        --fail \
        "$url" 2>/dev/null || echo "000")
    
    # Cache successful results
    if [ "$status_code" = "200" ]; then
        echo "$cache_key:200:$(date +%s)" >> "$CACHE_FILE"
        return 0
    else
        bmad_log "WARN" "External link failed: $url (HTTP $status_code) in $source_file"
        return 1
    fi
}

# Validate internal file reference
validate_internal_link() {
    local link="$1"
    local source_file="$2"
    
    # Remove leading ./
    link="${link#./}"
    
    # Handle relative paths
    if [[ ! "$link" =~ ^/ ]]; then
        local source_dir="$(dirname "$source_file")"
        link="$source_dir/$link"
    fi
    
    # Normalize path
    link="$(cd "$(dirname "$link")" 2>/dev/null && pwd)/$(basename "$link")" 2>/dev/null || echo "$link"
    
    if [ -f "$link" ] || [ -d "$link" ]; then
        return 0
    else
        bmad_log "ERROR" "Internal link broken: $link in $source_file"
        return 1
    fi
}

# Validate anchor reference
validate_anchor_link() {
    local file="$1"
    local anchor="$2"
    local source_file="$3"
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    # Convert anchor to expected heading format
    local expected_heading="$(echo "$anchor" | sed 's/-/ /g' | tr '[:upper:]' '[:lower:]')"
    
    # Check if heading exists in file
    if grep -qi "^#.*$expected_heading" "$file" 2>/dev/null; then
        return 0
    else
        bmad_log "WARN" "Anchor not found: #$anchor in $file (referenced from $source_file)"
        return 1
    fi
}

# Extract and validate links from a markdown file
validate_file_links() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    local total_links=0
    local broken_links=0
    local external_links=0
    local internal_links=0
    local anchor_links=0
    
    # Extract all markdown links
    while IFS= read -r line; do
        # Match [text](url) or [text](file#anchor)
        echo "$line" | grep -oE '\[[^]]*\]\([^)]+\)' | while IFS= read -r link; do
            local url="$(echo "$link" | sed -n 's/.*](\([^)]*\)).*/\1/p')"
            
            if [ -z "$url" ]; then
                continue
            fi
            
            ((total_links++))
            
            # Categorize and validate link
            if [[ "$url" =~ ^https?:// ]]; then
                # External URL
                ((external_links++))
                if [ "$VALIDATE_EXTERNAL" = true ]; then
                    if ! validate_external_url "$url" "$file"; then
                        ((broken_links++))
                        echo "❌ EXTERNAL: $url" >> "/tmp/bmad-broken-links.tmp"
                    fi
                fi
            elif [[ "$url" =~ ^mailto: ]] || [[ "$url" =~ ^ftp: ]]; then
                # Skip mailto and ftp links
                continue
            elif [[ "$url" =~ # ]]; then
                # Anchor link (internal file with anchor)
                ((anchor_links++))
                if [ "$VALIDATE_ANCHORS" = true ]; then
                    local target_file="${url%#*}"
                    local anchor="${url#*#}"
                    
                    if [ -z "$target_file" ]; then
                        target_file="$file"
                    fi
                    
                    if ! validate_anchor_link "$target_file" "$anchor" "$file"; then
                        ((broken_links++))
                        echo "⚠️ ANCHOR: $url" >> "/tmp/bmad-broken-links.tmp"
                    fi
                fi
            else
                # Internal file link
                ((internal_links++))
                if [ "$VALIDATE_INTERNAL" = true ]; then
                    if ! validate_internal_link "$url" "$file"; then
                        ((broken_links++))
                        echo "🔗 INTERNAL: $url" >> "/tmp/bmad-broken-links.tmp"
                    fi
                fi
            fi
        done
    done < "$file"
    
    # Store results for this file
    echo "$file:$total_links:$broken_links:$external_links:$internal_links:$anchor_links" >> "/tmp/bmad-link-results.tmp"
}

echo -e "${BLUE}🔗 BMAD Link Validation System${NC}"
echo "============================================================"
bmad_log "INFO" "Starting link validation scan"

# Check if running in BMAD context
if [ "$BMAD_ENABLED" = true ]; then
    echo -e "${PURPLE}🤖 BMAD Integration Active${NC}"
    echo "   Agent: $BMAD_AGENT"
    echo "   Project: $(basename "$BMAD_PROJECT_PATH")"
    bmad_log "INFO" "BMAD integration active for project $(basename "$BMAD_PROJECT_PATH")"
fi

# Clean temporary files
rm -f "/tmp/bmad-broken-links.tmp" "/tmp/bmad-link-results.tmp"

# Initialize validation report with BMAD header
cat > "$REPORT_FILE" << EOF
# BMAD Link Validation Report

**Generated**: $(date)  
**Agent**: $BMAD_AGENT  
**Scan Directory**: $(pwd)  
**BMAD Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ Active" || echo "❌ Disabled")

## Executive Summary
EOF

echo -e "\n${PURPLE}🔍 Scanning Documentation Files${NC}"
echo "--------------------------------------------"
bmad_log "INFO" "Beginning file discovery and link extraction"

# Find all markdown files
TOTAL_FILES=0
while IFS= read -r file; do
    ((TOTAL_FILES++))
    echo -e "  📄 Analyzing: $file"
    validate_file_links "$file" &
    
    # Limit parallel processes
    if (( $(jobs -r | wc -l) >= MAX_PARALLEL )); then
        wait -n
    fi
done < <(find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./.bmad-core/packages/*/docs/*")

# Wait for all background jobs to complete
wait

# Aggregate results
TOTAL_LINKS=0
TOTAL_BROKEN=0
TOTAL_EXTERNAL=0
TOTAL_INTERNAL=0
TOTAL_ANCHOR=0

if [ -f "/tmp/bmad-link-results.tmp" ]; then
    while IFS=':' read -r file links broken external internal anchor; do
        TOTAL_LINKS=$((TOTAL_LINKS + links))
        TOTAL_BROKEN=$((TOTAL_BROKEN + broken))
        TOTAL_EXTERNAL=$((TOTAL_EXTERNAL + external))
        TOTAL_INTERNAL=$((TOTAL_INTERNAL + internal))
        TOTAL_ANCHOR=$((TOTAL_ANCHOR + anchor))
    done < "/tmp/bmad-link-results.tmp"
fi

# Calculate health score
if [ $TOTAL_LINKS -gt 0 ]; then
    LINK_HEALTH_SCORE=$(( (TOTAL_LINKS - TOTAL_BROKEN) * 100 / TOTAL_LINKS ))
else
    LINK_HEALTH_SCORE=100
fi

# Determine health status
if [ $LINK_HEALTH_SCORE -ge 95 ]; then
    HEALTH_STATUS="🎉 EXCELLENT"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="SUCCESS"
elif [ $LINK_HEALTH_SCORE -ge 85 ]; then
    HEALTH_STATUS="✅ GOOD"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="INFO"
elif [ $LINK_HEALTH_SCORE -ge 70 ]; then
    HEALTH_STATUS="⚠️ ATTENTION NEEDED"
    HEALTH_COLOR="${YELLOW}"
    BMAD_ALERT_LEVEL="WARN"
else
    HEALTH_STATUS="❌ CRITICAL"
    HEALTH_COLOR="${RED}"
    BMAD_ALERT_LEVEL="CRITICAL"
fi

echo -e "\n${PURPLE}📊 Link Validation Results${NC}"
echo "-----------------------------------------"
echo -e "Total Files Scanned: $TOTAL_FILES"
echo -e "Total Links Found: $TOTAL_LINKS"
echo -e "  External Links: $TOTAL_EXTERNAL"
echo -e "  Internal Links: $TOTAL_INTERNAL"
echo -e "  Anchor Links: $TOTAL_ANCHOR"
echo -e "Broken Links: ${RED}$TOTAL_BROKEN${NC}"
echo -e "Link Health Score: ${HEALTH_COLOR}$LINK_HEALTH_SCORE%${NC} ($HEALTH_STATUS)"

bmad_log "$BMAD_ALERT_LEVEL" "Link validation completed. Health score: $LINK_HEALTH_SCORE% ($HEALTH_STATUS)"
bmad_notify "LINK_HEALTH_SCORE" "$LINK_HEALTH_SCORE:$HEALTH_STATUS:$TOTAL_BROKEN"

# Generate detailed BMAD report
cat >> "$REPORT_FILE" << EOF

## BMAD Link Health Analysis

**Overall Health Score**: $LINK_HEALTH_SCORE%  
**Status**: $HEALTH_STATUS  
**Alert Level**: $BMAD_ALERT_LEVEL

### Link Statistics
- **Total Files Scanned**: $TOTAL_FILES
- **Total Links Found**: $TOTAL_LINKS
- **External Links**: $TOTAL_EXTERNAL ($([ $TOTAL_LINKS -gt 0 ] && echo "$(( TOTAL_EXTERNAL * 100 / TOTAL_LINKS ))" || echo "0")%)
- **Internal Links**: $TOTAL_INTERNAL ($([ $TOTAL_LINKS -gt 0 ] && echo "$(( TOTAL_INTERNAL * 100 / TOTAL_LINKS ))" || echo "0")%)
- **Anchor Links**: $TOTAL_ANCHOR ($([ $TOTAL_LINKS -gt 0 ] && echo "$(( TOTAL_ANCHOR * 100 / TOTAL_LINKS ))" || echo "0")%)
- **Broken Links**: $TOTAL_BROKEN ($([ $TOTAL_LINKS -gt 0 ] && echo "$(( TOTAL_BROKEN * 100 / TOTAL_LINKS ))" || echo "0")%)

## Validation Configuration
- **External Link Validation**: $([ "$VALIDATE_EXTERNAL" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- **Internal Link Validation**: $([ "$VALIDATE_INTERNAL" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- **Anchor Link Validation**: $([ "$VALIDATE_ANCHORS" = true ] && echo "✅ Enabled" || echo "❌ Disabled")
- **Request Timeout**: ${TIMEOUT_SECONDS}s
- **Parallel Checks**: $MAX_PARALLEL concurrent

EOF

# Add broken links details if any exist
if [ $TOTAL_BROKEN -gt 0 ] && [ -f "/tmp/bmad-broken-links.tmp" ]; then
    cat >> "$REPORT_FILE" << EOF

## 🚨 Broken Links Details

$(cat "/tmp/bmad-broken-links.tmp" | sort | uniq | sed 's/^/- /')

### BMAD Repair Actions
Run these commands to interactively fix broken links:

\`\`\`bash
# Interactive link repair
bmad agent doc-health fix-broken-links

# Automatic link updates
bmad run validate-doc-links auto_fix=true

# Re-scan after fixes
bmad run validate-doc-links
\`\`\`

EOF
    bmad_notify "BROKEN_LINKS" "$TOTAL_BROKEN"
fi

# Add BMAD integration status
cat >> "$REPORT_FILE" << EOF

## BMAD Integration Status

- **Agent**: $BMAD_AGENT ✅ Active
- **Real-time Monitoring**: $([ -f "/tmp/bmad-link-health-cache" ] && echo "✅ Enabled" || echo "❌ Disabled")
- **Link Caching**: $([ -f "$CACHE_FILE" ] && echo "✅ Active" || echo "❌ Disabled")
- **Automated Repairs**: Available via \`bmad agent doc-health\`
- **Next Scheduled Check**: $(date -d '+1 week' 2>/dev/null || date -v +1w 2>/dev/null || echo "Configure with bmad schedule")

## Available BMAD Commands

\`\`\`bash
# Link validation
bmad run validate-doc-links                     # Full validation scan
bmad run validate-doc-links external_only=true # External links only
bmad run validate-doc-links internal_only=true # Internal links only

# Interactive repairs
bmad agent doc-health fix-broken-links         # Interactive link fixing
bmad agent doc-health update-redirects         # Handle redirected URLs

# Automation
bmad schedule validate-doc-links --weekly       # Weekly validation
bmad hook post-edit validate-doc-links         # Validate after edits
\`\`\`

---

**Generated by**: BMAD Link Validation System  
**Agent**: $BMAD_AGENT  
**Version**: 1.0.0  
**Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ BMAD Active" || echo "❌ BMAD Disabled")  
**Last Updated**: $(date)
EOF

# Update status line cache for real-time display
if [ "$BMAD_ENABLED" = true ]; then
    echo "$(date +%s):$LINK_HEALTH_SCORE%:$TOTAL_BROKEN" > /tmp/bmad-link-health-cache
    bmad_log "INFO" "Updated status line cache with link health: $LINK_HEALTH_SCORE%, $TOTAL_BROKEN broken"
fi

# Final summary
echo -e "\n${BLUE}🎯 BMAD Link Validation Complete${NC}"
echo "================================================"
echo -e "Link Health Score: ${HEALTH_COLOR}$LINK_HEALTH_SCORE%${NC} ($HEALTH_STATUS)"
echo -e "Agent: ${PURPLE}$BMAD_AGENT${NC}"
echo -e "Report saved: ${BLUE}$REPORT_FILE${NC}"

if [ $TOTAL_BROKEN -gt 0 ]; then
    echo -e "\n${RED}🔗 BROKEN: $TOTAL_BROKEN links need attention${NC}"
    echo -e "   Recommended: ${YELLOW}bmad agent doc-health fix-broken-links${NC}"
    bmad_notify "VALIDATION_CRITICAL" "$TOTAL_BROKEN:links_need_repair"
fi

if [ $TOTAL_BROKEN -gt 0 ] && [ $TOTAL_BROKEN -le 5 ]; then
    echo -e "\n${YELLOW}💡 BMAD Quick Actions:${NC}"
    echo "  - bmad agent doc-health fix-broken-links    # Interactive repair"
    echo "  - bmad run validate-doc-links auto_fix=true # Automated fixes"
fi

echo -e "\n${GREEN}🎯 BMAD Integration Status:${NC}"
echo "  🔗 Cache: $([ -f "$CACHE_FILE" ] && echo "Active ($(wc -l < "$CACHE_FILE") URLs cached)" || echo "Empty")"
echo "  📊 Status Line: Link health displayed in Claude Code"
echo "  🤖 Agent: Dr. DocHealth available for interactive repairs"
echo "  ⚡ Real-time: Link changes trigger automatic validation"

bmad_log "INFO" "Link validation completed. Health: $LINK_HEALTH_SCORE%, Broken: $TOTAL_BROKEN"
bmad_notify "VALIDATION_COMPLETE" "$LINK_HEALTH_SCORE:$HEALTH_STATUS:$TOTAL_BROKEN"

# Cleanup temporary files
rm -f "/tmp/bmad-broken-links.tmp" "/tmp/bmad-link-results.tmp"

# Set exit code based on validation results
if [ $TOTAL_BROKEN -gt 10 ]; then
    echo -e "\n${RED}⚠️ Exit Code 2: Too many broken links ($TOTAL_BROKEN)${NC}"
    bmad_log "ERROR" "Too many broken links found. Exiting with code 2"
    exit 2
elif [ $TOTAL_BROKEN -gt 0 ]; then
    echo -e "\n${YELLOW}⚠️ Exit Code 1: Broken links found ($TOTAL_BROKEN)${NC}"
    bmad_log "WARN" "Broken links found. Exiting with code 1"
    exit 1
else
    echo -e "\n${GREEN}✅ Exit Code 0: All links healthy${NC}"
    bmad_log "INFO" "All links healthy. Exiting with code 0"
    exit 0
fi
#!/bin/bash
# BMAD Documentation Maintenance System
# Enhanced maintenance and quality assurance for documentation with BMAD integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
MAX_FILE_SIZE=15000  # bytes (Claude Code compatibility)
LARGE_FILE_SIZE=10000  # bytes (warning threshold)
REPORT_FILE="docs/automation/bmad-maintenance-report.md"
BMAD_LOG_FILE="docs/automation/bmad-maintenance.log"

# BMAD Integration
BMAD_ENABLED=${BMAD_ENABLED:-true}
BMAD_AGENT="doc-health"
BMAD_PROJECT_PATH=${BMAD_PROJECT_PATH:-$(pwd)}

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
        # Notify BMAD system of important events
        echo "BMAD_EVENT:$event:$data" >> /tmp/bmad-doc-events.log
    fi
}

echo -e "${BLUE}🔧 BMAD Documentation Maintenance System${NC}"
echo "============================================================"
bmad_log "INFO" "Starting documentation maintenance scan"

# Check if running in BMAD context
if [ "$BMAD_ENABLED" = true ]; then
    echo -e "${PURPLE}🤖 BMAD Integration Active${NC}"
    echo "   Agent: $BMAD_AGENT"
    echo "   Project: $(basename "$BMAD_PROJECT_PATH")"
    bmad_log "INFO" "BMAD integration active for project $(basename "$BMAD_PROJECT_PATH")"
fi

# Initialize maintenance report with BMAD header
cat > "$REPORT_FILE" << EOF
# BMAD Documentation Maintenance Report

**Generated**: $(date)  
**Agent**: $BMAD_AGENT  
**Scan Directory**: $(pwd)  
**BMAD Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ Active" || echo "❌ Disabled")

## Executive Summary
EOF

# File size analysis with BMAD categorization
echo -e "\n${PURPLE}📊 File Size Analysis${NC}"
echo "------------------------"
bmad_log "INFO" "Beginning file size analysis"

TOTAL_FILES=0
OPTIMAL_FILES=0
LARGE_FILES=0
OVERSIZED_FILES=0
CRITICAL_FILES=0

# BMAD Health Categories
EXCELLENT_FILES=0
GOOD_FILES=0
WARNING_FILES=0
CRITICAL_HEALTH_FILES=0

declare -a CRITICAL_ISSUES=()
declare -a WARNINGS=()
declare -a RECOMMENDATIONS=()

find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./.bmad-core/packages/*/docs/*" | while read -r file; do
    
    size=$(wc -c < "$file")
    ((TOTAL_FILES++))
    
    # Size categorization
    if [ $size -gt 20000 ]; then
        echo -e "  ${RED}🚨 CRITICAL:${NC} $size bytes - $file"
        echo "- **🚨 CRITICAL**: \`$file\` ($size bytes) - BREAKS Claude Code compatibility" >> "$REPORT_FILE"
        ((CRITICAL_FILES++))
        CRITICAL_ISSUES+=("$file exceeds 20KB ($size bytes) - Immediate action required")
        bmad_log "CRITICAL" "File $file exceeds 20KB ($size bytes)"
    elif [ $size -gt $MAX_FILE_SIZE ]; then
        echo -e "  ${RED}❌ OVERSIZED:${NC} $size bytes - $file"
        echo "- **OVERSIZED**: \`$file\` ($size bytes)" >> "$REPORT_FILE"
        ((OVERSIZED_FILES++))
        WARNINGS+=("$file approaches size limit ($size bytes)")
        bmad_log "WARN" "File $file is oversized ($size bytes)"
    elif [ $size -gt $LARGE_FILE_SIZE ]; then
        echo -e "  ${YELLOW}⚠️  LARGE:${NC} $size bytes - $file"
        echo "- **Large**: \`$file\` ($size bytes)" >> "$REPORT_FILE"
        ((LARGE_FILES++))
        bmad_log "INFO" "File $file is large ($size bytes)"
    else
        echo -e "  ${GREEN}✅ OPTIMAL:${NC} $size bytes - $file"
        ((OPTIMAL_FILES++))
        bmad_log "DEBUG" "File $file is optimal size ($size bytes)"
    fi
done

echo -e "\n${BLUE}Size Distribution:${NC}"
echo "  Optimal (< 10KB): $OPTIMAL_FILES"
echo "  Large (10-15KB): $LARGE_FILES"
echo "  Oversized (> 15KB): $OVERSIZED_FILES"
echo "  Critical (> 20KB): $CRITICAL_FILES"

# BMAD Health Score Calculation
TOTAL_ISSUES=$((CRITICAL_FILES * 3 + OVERSIZED_FILES * 2 + LARGE_FILES * 1))
if [ $TOTAL_FILES -gt 0 ]; then
    # Advanced health scoring algorithm
    MAX_POSSIBLE_ISSUES=$((TOTAL_FILES * 3))
    HEALTH_SCORE=$(( (MAX_POSSIBLE_ISSUES - TOTAL_ISSUES) * 100 / MAX_POSSIBLE_ISSUES ))
else
    HEALTH_SCORE=100
fi

# Content quality checks with BMAD analysis
echo -e "\n${PURPLE}🔍 Content Quality Analysis${NC}"
echo "--------------------------------"
bmad_log "INFO" "Beginning content quality analysis"

MISSING_FRONTMATTER=0
EMPTY_FILES=0
NO_HEADINGS=0
BROKEN_STRUCTURE=0

find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.bmad-core/packages/*/docs/*" | while read -r file; do
    
    # Check for empty files
    if [ ! -s "$file" ]; then
        echo -e "  ${RED}📄 Empty file:${NC} $file"
        echo "- **Empty File**: \`$file\`" >> "$REPORT_FILE"
        ((EMPTY_FILES++))
        CRITICAL_ISSUES+=("Empty file: $file")
        bmad_log "WARN" "Empty file detected: $file"
        continue
    fi
    
    # Check for missing main heading
    if ! grep -q "^# " "$file"; then
        echo -e "  ${YELLOW}📝 No main heading:${NC} $file"
        echo "- **No Main Heading**: \`$file\`" >> "$REPORT_FILE"
        ((NO_HEADINGS++))
        WARNINGS+=("Missing H1 heading: $file")
        bmad_log "WARN" "No main heading in file: $file"
    fi
    
    # Check for proper heading structure
    local h1_count=$(grep -c "^# " "$file" 2>/dev/null || echo "0")
    if [ "$h1_count" -gt 1 ]; then
        echo -e "  ${YELLOW}📋 Multiple H1 headings:${NC} $file ($h1_count found)"
        echo "- **Multiple H1 Headings**: \`$file\` ($h1_count found)" >> "$REPORT_FILE"
        ((BROKEN_STRUCTURE++))
        WARNINGS+=("Multiple H1 headings: $file")
        bmad_log "WARN" "Multiple H1 headings in file: $file"
    fi
done

# BMAD Health Assessment
echo -e "\n${PURPLE}🏥 BMAD Health Assessment${NC}"
echo "------------------------------------"

# Health status determination with BMAD logic
if [ $HEALTH_SCORE -ge 95 ]; then
    HEALTH_STATUS="🎉 EXCELLENT"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="SUCCESS"
elif [ $HEALTH_SCORE -ge 85 ]; then
    HEALTH_STATUS="✅ GOOD"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="INFO"
elif [ $HEALTH_SCORE -ge 70 ]; then
    HEALTH_STATUS="⚠️ ATTENTION NEEDED"
    HEALTH_COLOR="${YELLOW}"
    BMAD_ALERT_LEVEL="WARN"
else
    HEALTH_STATUS="❌ CRITICAL"
    HEALTH_COLOR="${RED}"
    BMAD_ALERT_LEVEL="CRITICAL"
fi

echo -e "Health Score: ${HEALTH_COLOR}$HEALTH_SCORE%${NC} ($HEALTH_STATUS)"
bmad_log "$BMAD_ALERT_LEVEL" "Documentation health score: $HEALTH_SCORE% ($HEALTH_STATUS)"
bmad_notify "HEALTH_SCORE" "$HEALTH_SCORE:$HEALTH_STATUS"

# Generate detailed BMAD report
cat >> "$REPORT_FILE" << EOF

## BMAD Health Analysis

**Overall Health Score**: $HEALTH_SCORE%  
**Status**: $HEALTH_STATUS  
**Alert Level**: $BMAD_ALERT_LEVEL

### Detailed Metrics
- **Total Documentation Files**: $TOTAL_FILES
- **Optimal Files**: $OPTIMAL_FILES ($(( TOTAL_FILES > 0 ? OPTIMAL_FILES * 100 / TOTAL_FILES : 0 ))%)
- **Large Files**: $LARGE_FILES ($(( TOTAL_FILES > 0 ? LARGE_FILES * 100 / TOTAL_FILES : 0 ))%)
- **Oversized Files**: $OVERSIZED_FILES ($(( TOTAL_FILES > 0 ? OVERSIZED_FILES * 100 / TOTAL_FILES : 0 ))%)
- **Critical Files**: $CRITICAL_FILES ($(( TOTAL_FILES > 0 ? CRITICAL_FILES * 100 / TOTAL_FILES : 0 ))%)

### Quality Issues
- **Empty Files**: $EMPTY_FILES
- **Missing H1 Headings**: $NO_HEADINGS
- **Broken Structure**: $BROKEN_STRUCTURE

## BMAD Recommendations

EOF

# Generate BMAD-specific recommendations
if [ $CRITICAL_FILES -gt 0 ]; then
    cat >> "$REPORT_FILE" << EOF
### 🚨 Critical Actions Required
**Priority**: IMMEDIATE - These files break Claude Code compatibility

$(printf '%s\n' "${CRITICAL_ISSUES[@]}" | sed 's/^/- /')

**Recommended Actions**:
1. Use BMAD task: \`bmad run enforce-doc-standards auto_fix=true\`
2. Consider modular restructuring with \`bmad run generate-doc-sitemap\`
3. Apply enterprise templates: \`bmad apply template api-documentation\`

EOF
    bmad_notify "CRITICAL_FILES" "$CRITICAL_FILES"
fi

if [ $OVERSIZED_FILES -gt 0 ]; then
    cat >> "$REPORT_FILE" << EOF
### ⚠️ Size Warnings  
**Priority**: HIGH - Monitor for continued growth

$(printf '%s\n' "${WARNINGS[@]}" | sed 's/^/- /')

**BMAD Solutions**:
- \`bmad run check-doc-health detailed=true\` - Detailed analysis
- \`bmad agent doc-health split-oversized\` - Interactive splitting
- \`bmad run enforce-doc-standards restructure_content=true\` - Auto-restructure

EOF
    bmad_notify "OVERSIZED_FILES" "$OVERSIZED_FILES"
fi

# BMAD Integration Status
cat >> "$REPORT_FILE" << EOF

## BMAD Integration Status

- **Agent**: $BMAD_AGENT ✅ Active
- **Real-time Monitoring**: $([ -f "/tmp/claude-doc-health-cache" ] && echo "✅ Enabled" || echo "❌ Disabled")
- **Status Line Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ Active" || echo "❌ Disabled")
- **Automated Tasks**: Available via \`bmad run\` commands
- **Next Scheduled Check**: $(date -d '+1 month' 2>/dev/null || date -v +1m 2>/dev/null || echo "Configure with bmad schedule")

## Maintenance Actions Taken

$(date): BMAD automated maintenance scan completed
- Generated comprehensive health report with BMAD integration
- Identified $TOTAL_ISSUES total issues requiring attention
- Health score calculated: $HEALTH_SCORE% ($HEALTH_STATUS)
- BMAD recommendations generated
- Status line cache updated

## Available BMAD Commands

\`\`\`bash
# Health monitoring
bmad run check-doc-health                    # Comprehensive health check
bmad run validate-doc-links                  # Link validation
bmad run generate-doc-sitemap               # Navigation generation
bmad run enforce-doc-standards              # Standards compliance

# Agent interactions
bmad agent doc-health analyze              # Interactive analysis
bmad agent doc-health fix-critical         # Fix critical issues
bmad agent doc-health optimize-structure   # Structure optimization

# Scheduling
bmad schedule check-doc-health --weekly     # Weekly health checks
bmad schedule validate-doc-links --daily    # Daily link validation
\`\`\`

---

**Generated by**: BMAD Documentation Maintenance System  
**Agent**: $BMAD_AGENT  
**Version**: 1.0.0  
**Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ BMAD Active" || echo "❌ BMAD Disabled")  
**Last Updated**: $(date)
EOF

# Update status line cache for real-time display
if [ "$BMAD_ENABLED" = true ]; then
    echo "$(date +%s):$HEALTH_SCORE%" > /tmp/bmad-doc-health-cache
    bmad_log "INFO" "Updated status line cache with health score: $HEALTH_SCORE%"
fi

# Final summary with BMAD context
echo -e "\n${BLUE}📊 BMAD Maintenance Complete${NC}"
echo "=============================================="
echo -e "Health Score: ${HEALTH_COLOR}$HEALTH_SCORE%${NC} ($HEALTH_STATUS)"
echo -e "Agent: ${PURPLE}$BMAD_AGENT${NC}"
echo -e "Report saved: ${BLUE}$REPORT_FILE${NC}"

if [ $CRITICAL_FILES -gt 0 ]; then
    echo -e "\n${RED}🚨 CRITICAL: $CRITICAL_FILES files break Claude Code compatibility${NC}"
    echo -e "   Recommended: ${YELLOW}bmad run enforce-doc-standards auto_fix=true${NC}"
    bmad_notify "MAINTENANCE_CRITICAL" "$CRITICAL_FILES:files_need_immediate_attention"
fi

if [ $TOTAL_ISSUES -gt 0 ]; then
    echo -e "\n${YELLOW}💡 BMAD Actions Available:${NC}"
    echo "  - bmad agent doc-health analyze    # Interactive problem solving"
    echo "  - bmad run check-doc-health        # Detailed health analysis"
    echo "  - bmad run enforce-doc-standards   # Automated fixes"
    echo "  - bmad run validate-doc-links      # Link integrity check"
fi

echo -e "\n${GREEN}🎯 BMAD Integration Status:${NC}"
echo "  📚 Status Line: Health score displayed in Claude Code"
echo "  🤖 Agent: Dr. DocHealth available for interactive assistance"  
echo "  📋 Tasks: 4 automation tasks ready via 'bmad run'"
echo "  ⚡ Real-time: File changes trigger automatic health updates"

bmad_log "INFO" "Maintenance scan completed. Health score: $HEALTH_SCORE%"
bmad_notify "MAINTENANCE_COMPLETE" "$HEALTH_SCORE:$HEALTH_STATUS:$TOTAL_ISSUES"

# Set exit code based on health score for CI/CD integration
if [ $CRITICAL_FILES -gt 0 ]; then
    echo -e "\n${RED}⚠️ Exit Code 2: Critical issues found${NC}"
    bmad_log "ERROR" "Critical issues found. Exiting with code 2"
    exit 2
elif [ $HEALTH_SCORE -lt 70 ]; then
    echo -e "\n${YELLOW}⚠️ Exit Code 1: Health score below 70%${NC}"
    bmad_log "WARN" "Health score below threshold. Exiting with code 1"
    exit 1
else
    echo -e "\n${GREEN}✅ Exit Code 0: Documentation health acceptable${NC}"
    bmad_log "INFO" "Documentation health acceptable. Exiting with code 0"
    exit 0
fi
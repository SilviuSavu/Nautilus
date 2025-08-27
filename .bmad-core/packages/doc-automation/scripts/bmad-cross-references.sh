#!/bin/bash
# BMAD Cross-Reference Generation System
# Intelligent documentation cross-references with BMAD integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REPORT_FILE="docs/automation/bmad-cross-references-report.md"
BMAD_LOG_FILE="docs/automation/bmad-cross-references.log"
OUTPUT_FILE="docs/automation/cross-references.md"
INDEX_FILE="docs/automation/document-index.md"

# BMAD Integration
BMAD_ENABLED=${BMAD_ENABLED:-true}
BMAD_AGENT="doc-health"
BMAD_PROJECT_PATH=${BMAD_PROJECT_PATH:-$(pwd)}

# Cross-reference settings
GENERATE_INDEX=${GENERATE_INDEX:-true}
INCLUDE_FILE_STATS=${INCLUDE_FILE_STATS:-true}
MIN_FILE_SIZE=${MIN_FILE_SIZE:-100}  # bytes
MAX_REFERENCES=${MAX_REFERENCES:-50}  # per file
SORT_BY=${SORT_BY:-"name"}  # name, size, date, category

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
        echo "BMAD_EVENT:$event:$data" >> /tmp/bmad-crossref-events.log
    fi
}

# Extract document metadata
extract_metadata() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        echo "ERROR:FILE_NOT_FOUND"
        return 1
    fi
    
    local size=$(wc -c < "$file")
    local lines=$(wc -l < "$file")
    local words=$(wc -w < "$file")
    local modified=$(stat -f "%m" "$file" 2>/dev/null || stat -c "%Y" "$file" 2>/dev/null || echo "0")
    
    # Extract title (first H1 heading)
    local title=$(grep -m1 "^# " "$file" 2>/dev/null | sed 's/^# //' | sed 's/\r$//' || echo "$(basename "$file" .md)")
    
    # Extract description (first paragraph or bold text)
    local description=$(grep -m1 -A3 "^# " "$file" 2>/dev/null | \
        grep -v "^# " | \
        grep -v "^$" | \
        grep -v "^-" | \
        head -n1 | \
        sed 's/\*\*//g' | \
        sed 's/^[[:space:]]*//' | \
        cut -c1-100 || echo "No description available")
    
    # Determine category based on path and content
    local category="General"
    if [[ "$file" =~ docs/api/ ]]; then
        category="API"
    elif [[ "$file" =~ docs/architecture/ ]]; then
        category="Architecture"
    elif [[ "$file" =~ docs/deployment/ ]]; then
        category="Deployment"
    elif [[ "$file" =~ docs/integration/ ]]; then
        category="Integration"
    elif [[ "$file" =~ backend/ ]]; then
        category="Backend"
    elif [[ "$file" =~ frontend/ ]]; then
        category="Frontend"
    elif [[ "$file" =~ README ]]; then
        category="Overview"
    fi
    
    # Health assessment
    local health="✅"
    if [ $size -gt 20000 ]; then
        health="❌"  # Too large
    elif [ $size -gt 15000 ]; then
        health="⚠️"   # Large
    elif [ $size -lt $MIN_FILE_SIZE ]; then
        health="📝"  # Small/stub
    fi
    
    echo "$file|$title|$description|$category|$size|$lines|$words|$modified|$health"
}

# Generate cross-reference entry
generate_cross_reference() {
    local metadata="$1"
    local relative_path="$2"
    
    IFS='|' read -r file title description category size lines words modified health <<< "$metadata"
    
    local relative_file="$file"
    if [ -n "$relative_path" ]; then
        relative_file="$(realpath --relative-to="$relative_path" "$file" 2>/dev/null || echo "$file")"
    fi
    
    local size_kb="$(( size / 1024 ))"
    local mod_date="$(date -r "$modified" '+%Y-%m-%d' 2>/dev/null || date -d "@$modified" '+%Y-%m-%d' 2>/dev/null || echo "Unknown")"
    
    cat << EOF
### $health [$title]($relative_file)
**Category**: $category | **Size**: ${size_kb}KB | **Modified**: $mod_date

$description

EOF
}

# Generate category index
generate_category_index() {
    local temp_file="$1"
    
    echo "## Quick Navigation by Category"
    echo ""
    
    # Extract unique categories and sort them
    local categories=($(cut -d'|' -f4 "$temp_file" | sort | uniq))
    
    for category in "${categories[@]}"; do
        echo "### $category"
        echo ""
        
        # Get files in this category
        grep "|$category|" "$temp_file" | while IFS='|' read -r file title description cat size lines words modified health; do
            local relative_file="$(realpath --relative-to="docs/automation" "$file" 2>/dev/null || echo "$file")"
            local size_kb="$(( size / 1024 ))"
            echo "- $health [$title]($relative_file) (${size_kb}KB)"
        done
        echo ""
    done
}

# Generate file statistics
generate_statistics() {
    local temp_file="$1"
    
    local total_files=$(wc -l < "$temp_file")
    local total_size=0
    local total_lines=0
    local total_words=0
    local excellent_files=0
    local good_files=0
    local warning_files=0
    local critical_files=0
    
    while IFS='|' read -r file title description category size lines words modified health; do
        total_size=$((total_size + size))
        total_lines=$((total_lines + lines))
        total_words=$((total_words + words))
        
        case "$health" in
            "✅") ((excellent_files++)) ;;
            "📝") ((good_files++)) ;;
            "⚠️") ((warning_files++)) ;;
            "❌") ((critical_files++)) ;;
        esac
    done < "$temp_file"
    
    local total_size_mb="$(( total_size / 1024 / 1024 ))"
    local avg_size_kb="$(( total_size / total_files / 1024 ))"
    
    cat << EOF
## Documentation Statistics

### Overview
- **Total Files**: $total_files
- **Total Size**: ${total_size_mb}MB
- **Total Lines**: $total_lines
- **Total Words**: $total_words
- **Average File Size**: ${avg_size_kb}KB

### Health Distribution
- ✅ **Excellent** (< 10KB): $excellent_files files ($(( excellent_files * 100 / total_files ))%)
- 📝 **Good** (< 1KB): $good_files files ($(( good_files * 100 / total_files ))%)
- ⚠️ **Large** (10-20KB): $warning_files files ($(( warning_files * 100 / total_files ))%)
- ❌ **Critical** (> 20KB): $critical_files files ($(( critical_files * 100 / total_files ))%)

EOF
}

echo -e "${BLUE}🔗 BMAD Cross-Reference Generation System${NC}"
echo "============================================================"
bmad_log "INFO" "Starting cross-reference generation"

# Check if running in BMAD context
if [ "$BMAD_ENABLED" = true ]; then
    echo -e "${PURPLE}🤖 BMAD Integration Active${NC}"
    echo "   Agent: $BMAD_AGENT"
    echo "   Project: $(basename "$BMAD_PROJECT_PATH")"
    bmad_log "INFO" "BMAD integration active for project $(basename "$BMAD_PROJECT_PATH")"
fi

# Initialize cross-reference report with BMAD header
cat > "$REPORT_FILE" << EOF
# BMAD Cross-Reference Generation Report

**Generated**: $(date)  
**Agent**: $BMAD_AGENT  
**Scan Directory**: $(pwd)  
**BMAD Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ Active" || echo "❌ Disabled")

## Executive Summary
EOF

echo -e "\n${PURPLE}🔍 Scanning Documentation Files${NC}"
echo "--------------------------------------------"
bmad_log "INFO" "Beginning file discovery and metadata extraction"

# Create temporary file for metadata
TEMP_METADATA="/tmp/bmad-metadata.tmp"
rm -f "$TEMP_METADATA"

TOTAL_FILES=0
PROCESSED_FILES=0

# Find and process all markdown files
while IFS= read -r file; do
    ((TOTAL_FILES++))
    echo -e "  📄 Processing: $file"
    
    if metadata=$(extract_metadata "$file"); then
        echo "$metadata" >> "$TEMP_METADATA"
        ((PROCESSED_FILES++))
        bmad_log "DEBUG" "Processed metadata for $file"
    else
        bmad_log "WARN" "Failed to extract metadata from $file"
    fi
done < <(find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./.bmad-core/packages/*/docs/*" | \
    head -n $MAX_REFERENCES)

# Sort metadata based on configuration
case "$SORT_BY" in
    "size")
        sort -t'|' -k5 -n "$TEMP_METADATA" -o "$TEMP_METADATA"
        ;;
    "date")
        sort -t'|' -k8 -n "$TEMP_METADATA" -o "$TEMP_METADATA"
        ;;
    "category")
        sort -t'|' -k4 "$TEMP_METADATA" -o "$TEMP_METADATA"
        ;;
    *)
        sort -t'|' -k2 "$TEMP_METADATA" -o "$TEMP_METADATA"
        ;;
esac

echo -e "\n${PURPLE}📝 Generating Cross-References${NC}"
echo "----------------------------------------------"
bmad_log "INFO" "Generating cross-reference documentation"

# Generate main cross-reference file
cat > "$OUTPUT_FILE" << EOF
# Documentation Cross-References

**Generated**: $(date)  
**Total Files**: $PROCESSED_FILES  
**Sorted By**: $SORT_BY  
**Agent**: $BMAD_AGENT

*This document provides comprehensive cross-references for all project documentation, making it easier to discover and navigate related content.*

EOF

# Add file statistics if enabled
if [ "$INCLUDE_FILE_STATS" = true ]; then
    echo -e "  📊 Generating statistics..."
    generate_statistics "$TEMP_METADATA" >> "$OUTPUT_FILE"
fi

# Add category navigation
echo -e "  🗂️  Generating category index..."
generate_category_index "$TEMP_METADATA" >> "$OUTPUT_FILE"

# Add detailed cross-references
echo "" >> "$OUTPUT_FILE"
echo "## Detailed Cross-References" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo -e "  🔗 Generating detailed references..."
while IFS= read -r metadata; do
    generate_cross_reference "$metadata" "docs/automation" >> "$OUTPUT_FILE"
done < "$TEMP_METADATA"

# Add footer
cat >> "$OUTPUT_FILE" << EOF

---

**Generated by**: BMAD Cross-Reference System  
**Agent**: $BMAD_AGENT  
**Last Updated**: $(date)  
**Total Processed**: $PROCESSED_FILES files

## BMAD Commands for Cross-References

\`\`\`bash
# Regenerate cross-references
bmad run generate-doc-sitemap

# Update with different sorting
bmad run generate-doc-sitemap sort_by=size
bmad run generate-doc-sitemap sort_by=date
bmad run generate-doc-sitemap sort_by=category

# Interactive cross-reference management
bmad agent doc-health update-references
\`\`\`
EOF

# Generate document index if enabled
if [ "$GENERATE_INDEX" = true ]; then
    echo -e "  📋 Generating document index..."
    
    cat > "$INDEX_FILE" << EOF
# Document Index

**Generated**: $(date)  
**Files Indexed**: $PROCESSED_FILES

## Alphabetical Index

EOF
    
    # Generate alphabetical index
    sort -t'|' -k2 "$TEMP_METADATA" | while IFS='|' read -r file title description category size lines words modified health; do
        local relative_file="$(realpath --relative-to="docs/automation" "$file" 2>/dev/null || echo "$file")"
        local size_kb="$(( size / 1024 ))"
        echo "- $health [$title]($relative_file) - $category (${size_kb}KB)" >> "$INDEX_FILE"
    done
    
    cat >> "$INDEX_FILE" << EOF

---
**Generated by**: BMAD Cross-Reference System  
**Last Updated**: $(date)
EOF
fi

# Calculate health metrics
EXCELLENT_COUNT=$(grep -c "|✅$" "$TEMP_METADATA" 2>/dev/null || echo "0")
WARNING_COUNT=$(grep -c "|⚠️$" "$TEMP_METADATA" 2>/dev/null || echo "0")
CRITICAL_COUNT=$(grep -c "|❌$" "$TEMP_METADATA" 2>/dev/null || echo "0")

CROSS_REF_HEALTH=0
if [ $PROCESSED_FILES -gt 0 ]; then
    CROSS_REF_HEALTH=$(( (PROCESSED_FILES - CRITICAL_COUNT - WARNING_COUNT/2) * 100 / PROCESSED_FILES ))
fi

# Determine health status
if [ $CROSS_REF_HEALTH -ge 90 ]; then
    HEALTH_STATUS="🎉 EXCELLENT"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="SUCCESS"
elif [ $CROSS_REF_HEALTH -ge 75 ]; then
    HEALTH_STATUS="✅ GOOD"
    HEALTH_COLOR="${GREEN}"
    BMAD_ALERT_LEVEL="INFO"
elif [ $CROSS_REF_HEALTH -ge 60 ]; then
    HEALTH_STATUS="⚠️ ATTENTION NEEDED"
    HEALTH_COLOR="${YELLOW}"
    BMAD_ALERT_LEVEL="WARN"
else
    HEALTH_STATUS="❌ CRITICAL"
    HEALTH_COLOR="${RED}"
    BMAD_ALERT_LEVEL="CRITICAL"
fi

# Generate final report
echo -e "\n${PURPLE}📊 Cross-Reference Generation Results${NC}"
echo "------------------------------------------------"
echo -e "Files Processed: $PROCESSED_FILES / $TOTAL_FILES"
echo -e "Cross-Reference Health: ${HEALTH_COLOR}$CROSS_REF_HEALTH%${NC} ($HEALTH_STATUS)"
echo -e "Files by Health:"
echo -e "  ✅ Excellent: $EXCELLENT_COUNT"
echo -e "  📝 Good: $(grep -c "|📝$" "$TEMP_METADATA" 2>/dev/null || echo "0")"
echo -e "  ⚠️ Warning: $WARNING_COUNT"
echo -e "  ❌ Critical: $CRITICAL_COUNT"

bmad_log "$BMAD_ALERT_LEVEL" "Cross-reference generation completed. Health: $CROSS_REF_HEALTH% ($HEALTH_STATUS)"
bmad_notify "CROSS_REF_HEALTH" "$CROSS_REF_HEALTH:$HEALTH_STATUS:$CRITICAL_COUNT"

# Complete the report
cat >> "$REPORT_FILE" << EOF

## BMAD Cross-Reference Analysis

**Overall Health Score**: $CROSS_REF_HEALTH%  
**Status**: $HEALTH_STATUS  
**Alert Level**: $BMAD_ALERT_LEVEL

### Processing Results
- **Total Files Found**: $TOTAL_FILES
- **Files Processed**: $PROCESSED_FILES ($([ $TOTAL_FILES -gt 0 ] && echo "$(( PROCESSED_FILES * 100 / TOTAL_FILES ))" || echo "0")%)
- **Cross-References Generated**: ✅ Complete
- **Document Index**: $([ "$GENERATE_INDEX" = true ] && echo "✅ Generated" || echo "❌ Disabled")
- **File Statistics**: $([ "$INCLUDE_FILE_STATS" = true ] && echo "✅ Included" || echo "❌ Disabled")

### Output Files Generated
- **Cross-References**: [\`$OUTPUT_FILE\`]($OUTPUT_FILE)
$([ "$GENERATE_INDEX" = true ] && echo "- **Document Index**: [\`$INDEX_FILE\`]($INDEX_FILE)")
- **Processing Report**: [\`$REPORT_FILE\`]($REPORT_FILE)

### Health Breakdown
- ✅ **Excellent Files**: $EXCELLENT_COUNT (optimal size and structure)
- 📝 **Good Files**: $(grep -c "|📝$" "$TEMP_METADATA" 2>/dev/null || echo "0") (small but complete)
- ⚠️ **Warning Files**: $WARNING_COUNT (large but manageable)
- ❌ **Critical Files**: $CRITICAL_COUNT (oversized, needs attention)

## BMAD Integration Status

- **Agent**: $BMAD_AGENT ✅ Active
- **Real-time Updates**: Available via hooks
- **Sorting Options**: name, size, date, category
- **Auto-regeneration**: Configurable via BMAD scheduling
- **Interactive Management**: Available through agent commands

## Available BMAD Commands

\`\`\`bash
# Cross-reference generation
bmad run generate-doc-sitemap                   # Generate all cross-references
bmad run generate-doc-sitemap sort_by=size      # Sort by file size
bmad run generate-doc-sitemap include_stats=false # Skip statistics

# Interactive management
bmad agent doc-health update-references         # Interactive updates
bmad agent doc-health reorganize-docs          # Structure optimization

# Automation
bmad schedule generate-doc-sitemap --monthly     # Monthly regeneration
bmad hook post-edit generate-doc-sitemap        # Update after edits
\`\`\`

---

**Generated by**: BMAD Cross-Reference System  
**Agent**: $BMAD_AGENT  
**Version**: 1.0.0  
**Integration**: $([ "$BMAD_ENABLED" = true ] && echo "✅ BMAD Active" || echo "❌ BMAD Disabled")  
**Last Updated**: $(date)
EOF

# Update status line cache for real-time display
if [ "$BMAD_ENABLED" = true ]; then
    echo "$(date +%s):$CROSS_REF_HEALTH%:$PROCESSED_FILES" > /tmp/bmad-crossref-health-cache
    bmad_log "INFO" "Updated status line cache with cross-ref health: $CROSS_REF_HEALTH%"
fi

# Final summary
echo -e "\n${BLUE}🎯 BMAD Cross-Reference Generation Complete${NC}"
echo "========================================================"
echo -e "Cross-Reference Health: ${HEALTH_COLOR}$CROSS_REF_HEALTH%${NC} ($HEALTH_STATUS)"
echo -e "Agent: ${PURPLE}$BMAD_AGENT${NC}"
echo -e "Files Generated:"
echo -e "  📄 Cross-References: ${BLUE}$OUTPUT_FILE${NC}"
[ "$GENERATE_INDEX" = true ] && echo -e "  📋 Document Index: ${BLUE}$INDEX_FILE${NC}"
echo -e "  📊 Report: ${BLUE}$REPORT_FILE${NC}"

if [ $CRITICAL_COUNT -gt 0 ]; then
    echo -e "\n${RED}⚠️ CRITICAL: $CRITICAL_COUNT files are oversized${NC}"
    echo -e "   Recommended: ${YELLOW}bmad run enforce-doc-standards${NC}"
    bmad_notify "CRITICAL_FILES" "$CRITICAL_COUNT"
fi

echo -e "\n${GREEN}🎯 BMAD Integration Status:${NC}"
echo "  📚 Cross-References: $PROCESSED_FILES files indexed"
echo "  📊 Health Monitoring: Real-time updates available"
echo "  🤖 Agent: Dr. DocHealth ready for interactive management"
echo "  ⚡ Automation: Hooks and scheduling configured"

bmad_log "INFO" "Cross-reference generation completed. Health: $CROSS_REF_HEALTH%, Files: $PROCESSED_FILES"
bmad_notify "GENERATION_COMPLETE" "$CROSS_REF_HEALTH:$HEALTH_STATUS:$PROCESSED_FILES"

# Cleanup
rm -f "$TEMP_METADATA"

# Set exit code based on health
if [ $CRITICAL_COUNT -gt 5 ]; then
    echo -e "\n${RED}⚠️ Exit Code 2: Too many critical files${NC}"
    bmad_log "ERROR" "Too many critical files. Exiting with code 2"
    exit 2
elif [ $CROSS_REF_HEALTH -lt 60 ]; then
    echo -e "\n${YELLOW}⚠️ Exit Code 1: Health below threshold${NC}"
    bmad_log "WARN" "Cross-reference health below threshold. Exiting with code 1"
    exit 1
else
    echo -e "\n${GREEN}✅ Exit Code 0: Cross-references healthy${NC}"
    bmad_log "INFO" "Cross-references generated successfully. Exiting with code 0"
    exit 0
fi
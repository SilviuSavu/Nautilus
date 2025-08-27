#!/bin/bash
# Nautilus Documentation Maintenance System
# Automated maintenance and quality assurance for documentation

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
REPORT_FILE="docs/automation/maintenance-report.md"

echo -e "${BLUE}🔧 Nautilus Documentation Maintenance System${NC}"
echo "============================================================"

# Initialize maintenance report
cat > "$REPORT_FILE" << EOF
# Documentation Maintenance Report

**Generated**: $(date)  
**Scan Directory**: $(pwd)

## Maintenance Summary
EOF

# File size analysis
echo -e "\n${PURPLE}📊 File Size Analysis${NC}"
echo "------------------------"

TOTAL_FILES=0
OPTIMAL_FILES=0
LARGE_FILES=0
OVERSIZED_FILES=0

find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" | while read -r file; do
    
    size=$(wc -c < "$file")
    ((TOTAL_FILES++))
    
    if [ $size -gt $MAX_FILE_SIZE ]; then
        echo -e "  ${RED}❌ OVERSIZED:${NC} $size bytes - $file"
        echo "- **OVERSIZED**: \`$file\` ($size bytes)" >> "$REPORT_FILE"
        ((OVERSIZED_FILES++))
    elif [ $size -gt $LARGE_FILE_SIZE ]; then
        echo -e "  ${YELLOW}⚠️  LARGE:${NC} $size bytes - $file"
        echo "- **Large**: \`$file\` ($size bytes)" >> "$REPORT_FILE"
        ((LARGE_FILES++))
    else
        echo -e "  ${GREEN}✅ OPTIMAL:${NC} $size bytes - $file"
        ((OPTIMAL_FILES++))
    fi
done

echo -e "\n${BLUE}Size Distribution:${NC}"
echo "  Optimal (< 10KB): $OPTIMAL_FILES"
echo "  Large (10-15KB): $LARGE_FILES"
echo "  Oversized (> 15KB): $OVERSIZED_FILES"

# Content quality checks
echo -e "\n${PURPLE}🔍 Content Quality Analysis${NC}"
echo "--------------------------------"

MISSING_FRONTMATTER=0
EMPTY_FILES=0
NO_HEADINGS=0

find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" | while read -r file; do
    
    # Check for empty files
    if [ ! -s "$file" ]; then
        echo -e "  ${RED}📄 Empty file:${NC} $file"
        echo "- **Empty File**: \`$file\`" >> "$REPORT_FILE"
        ((EMPTY_FILES++))
        continue
    fi
    
    # Check for missing main heading
    if ! grep -q "^# " "$file"; then
        echo -e "  ${YELLOW}📝 No main heading:${NC} $file"
        echo "- **No Main Heading**: \`$file\`" >> "$REPORT_FILE"
        ((NO_HEADINGS++))
    fi
done

# Generate file structure report
echo -e "\n${PURPLE}📁 Documentation Structure${NC}"
echo "--------------------------------"

cat >> "$REPORT_FILE" << EOF

## File Structure Analysis

\`\`\`
EOF

# Create tree-like structure
find docs -name "*.md" -type f | sort | sed 's|docs/||' | while read -r file; do
    depth=$(echo "$file" | tr -cd '/' | wc -c)
    indent=$(printf "%*s" $((depth * 2)) "")
    filename=$(basename "$file")
    echo "$indent- $filename" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF
\`\`\`

## Quality Metrics

- **Total Files**: $TOTAL_FILES
- **Optimal Size**: $OPTIMAL_FILES
- **Large Files**: $LARGE_FILES  
- **Oversized Files**: $OVERSIZED_FILES
- **Empty Files**: $EMPTY_FILES
- **Missing Headings**: $NO_HEADINGS

## Recommendations

EOF

# Generate recommendations based on findings
if [ $OVERSIZED_FILES -gt 0 ]; then
    echo "### 🚨 Critical: Oversized Files" >> "$REPORT_FILE"
    echo "Files exceeding 15KB break Claude Code compatibility. Consider splitting these files:" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

if [ $LARGE_FILES -gt 0 ]; then
    echo "### ⚠️ Warning: Large Files" >> "$REPORT_FILE"
    echo "Files between 10-15KB should be monitored for growth. Consider splitting if they continue to grow:" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

if [ $EMPTY_FILES -gt 0 ]; then
    echo "### 📄 Empty Files" >> "$REPORT_FILE"
    echo "Remove or populate empty documentation files:" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Documentation health score
TOTAL_ISSUES=$((OVERSIZED_FILES + LARGE_FILES + EMPTY_FILES + NO_HEADINGS))
if [ $TOTAL_FILES -gt 0 ]; then
    HEALTH_SCORE=$(( (TOTAL_FILES - TOTAL_ISSUES) * 100 / TOTAL_FILES ))
else
    HEALTH_SCORE=100
fi

echo -e "\n${PURPLE}🏥 Documentation Health Score${NC}"
echo "------------------------------------"
echo -e "Health Score: ${GREEN}$HEALTH_SCORE%${NC}"

if [ $HEALTH_SCORE -ge 90 ]; then
    echo -e "Status: ${GREEN}Excellent${NC} 🎉"
    HEALTH_STATUS="Excellent 🎉"
elif [ $HEALTH_SCORE -ge 75 ]; then
    echo -e "Status: ${YELLOW}Good${NC} ✅"
    HEALTH_STATUS="Good ✅"
elif [ $HEALTH_SCORE -ge 60 ]; then
    echo -e "Status: ${YELLOW}Needs Attention${NC} ⚠️"
    HEALTH_STATUS="Needs Attention ⚠️"
else
    echo -e "Status: ${RED}Poor${NC} ❌"
    HEALTH_STATUS="Poor ❌"
fi

cat >> "$REPORT_FILE" << EOF

## Documentation Health Score

**Score**: $HEALTH_SCORE%  
**Status**: $HEALTH_STATUS

### Health Breakdown
- Total Documentation Files: $TOTAL_FILES
- Files with Issues: $TOTAL_ISSUES
- Healthy Files: $((TOTAL_FILES - TOTAL_ISSUES))

## Maintenance Actions Taken

$(date): Automated maintenance scan completed
- Generated maintenance report
- Identified $TOTAL_ISSUES issues requiring attention
- Health score calculated: $HEALTH_SCORE%

## Next Maintenance

**Recommended**: Run monthly maintenance scan
**Command**: \`./docs/automation/doc-maintenance.sh\`
**Link Validation**: \`./docs/automation/link-validator.sh\`

---

**Generated by**: Nautilus Documentation Maintenance System  
**Version**: 1.0  
**Last Updated**: $(date)
EOF

# Cleanup suggestions
echo -e "\n${PURPLE}🧹 Cleanup Suggestions${NC}"
echo "------------------------------------"

# Check for old temporary files
TEMP_FILES=$(find . -name "*.tmp" -o -name "*.bak" -o -name "*~" | wc -l)
if [ $TEMP_FILES -gt 0 ]; then
    echo -e "  Found $TEMP_FILES temporary files that can be cleaned up"
    find . -name "*.tmp" -o -name "*.bak" -o -name "*~" | head -5
fi

# Check for duplicate files (by name)
echo -e "\n${BLUE}📋 Duplicate File Names:${NC}"
find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" | \
    xargs basename -a | sort | uniq -d | while read -r duplicate; do
    echo -e "  ${YELLOW}⚠️  Duplicate name:${NC} $duplicate"
    find . -name "$duplicate" -type f \
        -not -path "./docs/archive/*" \
        -not -path "./node_modules/*"
done

# Final summary
echo -e "\n${BLUE}📊 Maintenance Complete${NC}"
echo "======================================"
echo -e "Health Score: ${GREEN}$HEALTH_SCORE%${NC} ($HEALTH_STATUS)"
echo -e "Report saved: ${BLUE}$REPORT_FILE${NC}"

if [ $TOTAL_ISSUES -gt 0 ]; then
    echo -e "\n${YELLOW}💡 Action Required:${NC}"
    echo "  - $OVERSIZED_FILES oversized files need splitting"
    echo "  - $LARGE_FILES large files should be monitored"
    echo "  - $EMPTY_FILES empty files should be addressed"
    echo "  - $NO_HEADINGS files need proper headings"
fi

echo -e "\n${GREEN}🎯 Recommendations:${NC}"
echo "  1. Run link validation: ./docs/automation/link-validator.sh"
echo "  2. Review maintenance report: $REPORT_FILE"
echo "  3. Schedule monthly maintenance runs"
echo "  4. Address any oversized files immediately"

# Set exit code based on critical issues
if [ $OVERSIZED_FILES -gt 0 ]; then
    echo -e "\n${RED}⚠️  Exit Code 1: Critical issues found${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ Exit Code 0: No critical issues${NC}"
    exit 0
fi
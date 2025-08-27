#!/bin/bash
# Nautilus Documentation Link Validator
# Automatically validates all markdown links in the documentation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_LINKS=0
VALID_LINKS=0
BROKEN_LINKS=0
EXTERNAL_LINKS=0

# Output file for broken links
BROKEN_LINKS_FILE="docs/automation/broken-links-report.md"

echo -e "${BLUE}🔗 Nautilus Documentation Link Validator${NC}"
echo "====================================================="

# Initialize report file
cat > "$BROKEN_LINKS_FILE" << EOF
# Broken Links Report

**Generated**: $(date)  
**Scan Directory**: $(pwd)

## Summary
EOF

# Function to check if URL is accessible
check_url() {
    local url="$1"
    local file="$2"
    local line="$3"
    
    # Skip anchor-only links
    if [[ "$url" =~ ^#.* ]]; then
        return 0
    fi
    
    # Handle relative file links
    if [[ "$url" =~ ^[^http].* && ! "$url" =~ ^# ]]; then
        # Convert relative path to absolute
        local dir=$(dirname "$file")
        local full_path
        
        if [[ "$url" =~ ^/ ]]; then
            # Absolute path from project root
            full_path="${url#/}"
        else
            # Relative path from current file
            full_path="$dir/$url"
        fi
        
        # Remove any ../ and normalize path
        full_path=$(realpath -m "$full_path" 2>/dev/null || echo "$full_path")
        
        if [[ -f "$full_path" ]]; then
            echo -e "  ✅ $url"
            ((VALID_LINKS++))
            return 0
        else
            echo -e "  ${RED}❌ $url${NC} (file not found: $full_path)"
            echo "- **File**: $file:$line" >> "$BROKEN_LINKS_FILE"
            echo "  **Broken Link**: \`$url\`" >> "$BROKEN_LINKS_FILE"
            echo "  **Expected Path**: \`$full_path\`" >> "$BROKEN_LINKS_FILE"
            echo "" >> "$BROKEN_LINKS_FILE"
            ((BROKEN_LINKS++))
            return 1
        fi
    fi
    
    # Handle HTTP/HTTPS URLs
    if [[ "$url" =~ ^https?:// ]]; then
        ((EXTERNAL_LINKS++))
        
        # Check if URL is accessible (with timeout)
        if curl -s --head --request GET --max-time 10 "$url" > /dev/null 2>&1; then
            echo -e "  ✅ $url"
            ((VALID_LINKS++))
            return 0
        else
            echo -e "  ${RED}❌ $url${NC} (not accessible)"
            echo "- **File**: $file:$line" >> "$BROKEN_LINKS_FILE"
            echo "  **Broken Link**: \`$url\`" >> "$BROKEN_LINKS_FILE"
            echo "  **Error**: URL not accessible" >> "$BROKEN_LINKS_FILE"
            echo "" >> "$BROKEN_LINKS_FILE"
            ((BROKEN_LINKS++))
            return 1
        fi
    fi
    
    return 0
}

# Function to extract and validate links from a markdown file
validate_file_links() {
    local file="$1"
    
    echo -e "\n${BLUE}📄 Checking: $file${NC}"
    
    # Extract markdown links using grep and process line by line
    local line_num=0
    while IFS= read -r line; do
        ((line_num++))
        
        # Extract markdown links [text](url) from the line
        echo "$line" | grep -oE '\[[^]]*\]\([^)]+\)' | while read -r link; do
            # Extract URL from markdown link
            local url=$(echo "$link" | sed -n 's/.*\](\([^)]*\)).*/\1/p')
            
            if [[ -n "$url" ]]; then
                ((TOTAL_LINKS++))
                echo -e "  ${YELLOW}🔍 Line $line_num:${NC} $url"
                check_url "$url" "$file" "$line_num"
            fi
        done
    done < "$file"
}

# Find all markdown files (excluding archive and node_modules)
echo -e "\n${BLUE}🔍 Scanning for markdown files...${NC}"

find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./*/node_modules/*" \
    | sort | while read -r file; do
    validate_file_links "$file"
done

# Generate final report
echo -e "\n${BLUE}📊 Validation Summary${NC}"
echo "====================================================="
echo -e "Total Links Found: ${YELLOW}$TOTAL_LINKS${NC}"
echo -e "Valid Links: ${GREEN}$VALID_LINKS${NC}"
echo -e "Broken Links: ${RED}$BROKEN_LINKS${NC}"
echo -e "External Links: ${BLUE}$EXTERNAL_LINKS${NC}"

# Append summary to report
cat >> "$BROKEN_LINKS_FILE" << EOF

- **Total Links**: $TOTAL_LINKS
- **Valid Links**: $VALID_LINKS  
- **Broken Links**: $BROKEN_LINKS
- **External Links**: $EXTERNAL_LINKS

## Broken Links Details

EOF

if [ $BROKEN_LINKS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All links are valid!${NC}"
    echo "**Result**: ✅ All links are valid!" >> "$BROKEN_LINKS_FILE"
else
    echo -e "\n${RED}⚠️  Found $BROKEN_LINKS broken links. Check $BROKEN_LINKS_FILE for details.${NC}"
    echo -e "\n${YELLOW}💡 Run this script regularly to maintain link integrity.${NC}"
fi

echo -e "\n${BLUE}📋 Report saved to: $BROKEN_LINKS_FILE${NC}"

# Exit with error code if broken links found
exit $BROKEN_LINKS
#!/bin/bash
# Nautilus Cross-Reference Generator
# Automatically generates and updates cross-references between documentation files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
CROSS_REF_FILE="docs/automation/cross-references.md"
SITEMAP_FILE="docs/automation/sitemap.md"

echo -e "${BLUE}🔗 Nautilus Cross-Reference Generator${NC}"
echo "=================================================="

# Initialize cross-reference file
cat > "$CROSS_REF_FILE" << EOF
# Documentation Cross-References

**Generated**: $(date)  
**Purpose**: Automated cross-references between all documentation files

## File Relationships

EOF

# Initialize sitemap
cat > "$SITEMAP_FILE" << EOF
# Documentation Sitemap

**Generated**: $(date)  
**Total Files**: $(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | wc -l)

## Site Structure

EOF

# Initialize arrays for different systems
file_topics=()
file_descriptions=()
engine_files=()
api_files=()
architecture_files=()

# Function to extract topic keywords from a file
extract_topics() {
    local file="$1"
    local topics=""
    
    # Extract headings and common keywords
    grep -h "^#" "$file" 2>/dev/null | sed 's/^#* *//' | tr '[:upper:]' '[:lower:]' | \
    grep -o -E '\b(engine|api|architecture|performance|deployment|docker|database|risk|trading|portfolio|monitoring|testing|integration)\b' | \
    sort | uniq | tr '\n' ' '
}

# Function to extract file description from first paragraph
extract_description() {
    local file="$1"
    
    # Get first non-empty line after title that contains actual content
    sed -n '/^# /,/^$/p' "$file" 2>/dev/null | \
    grep -v '^#' | \
    grep -v '^$' | \
    head -1 | \
    sed 's/\*\*//g' | \
    sed 's/`//g' | \
    cut -c1-100
}

# Scan all markdown files
echo -e "\n${PURPLE}🔍 Scanning documentation files...${NC}"

while IFS= read -r -d '' file; do
    relative_path=${file#./}
    echo "  📄 Processing: $relative_path"
    
    # Extract topics and description
    topics=$(extract_topics "$file")
    description=$(extract_description "$file")
    
    file_topics["$relative_path"]="$topics"
    file_descriptions["$relative_path"]="$description"
    
    # Categorize files
    if [[ "$relative_path" =~ engine|ENGINE ]]; then
        engine_files["$relative_path"]=1
    fi
    
    if [[ "$relative_path" =~ api|API ]]; then
        api_files["$relative_path"]=1
    fi
    
    if [[ "$relative_path" =~ architecture|ARCHITECTURE ]]; then
        architecture_files["$relative_path"]=1
    fi
    
done < <(find . -name "*.md" -type f \
    -not -path "./docs/archive/*" \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -print0 | sort -z)

# Generate cross-references by category
echo -e "\n${PURPLE}🔗 Generating cross-references...${NC}"

# Engine-related files
if [ ${#engine_files[@]} -gt 0 ]; then
    cat >> "$CROSS_REF_FILE" << EOF
### Engine Documentation
$(for file in "${!engine_files[@]}"; do
    echo "- **[$file]($file)** - ${file_descriptions[$file]}"
done)

EOF
fi

# API-related files
if [ ${#api_files[@]} -gt 0 ]; then
    cat >> "$CROSS_REF_FILE" << EOF
### API Documentation
$(for file in "${!api_files[@]}"; do
    echo "- **[$file]($file)** - ${file_descriptions[$file]}"
done)

EOF
fi

# Architecture-related files
if [ ${#architecture_files[@]} -gt 0 ]; then
    cat >> "$CROSS_REF_FILE" << EOF
### Architecture Documentation
$(for file in "${!architecture_files[@]}"; do
    echo "- **[$file]($file)** - ${file_descriptions[$file]}"
done)

EOF
fi

# Generate topic-based cross-references
echo -e "\n${PURPLE}📋 Generating topic index...${NC}"

cat >> "$CROSS_REF_FILE" << EOF
## Topic Index

EOF

# Create topic index
topic_files=()

for file in "${!file_topics[@]}"; do
    for topic in ${file_topics[$file]}; do
        if [ -n "$topic" ]; then
            topic_files["$topic"]+="$file "
        fi
    done
done

# Output topic index
for topic in $(printf '%s\n' "${!topic_files[@]}" | sort); do
    echo "### $topic" >> "$CROSS_REF_FILE"
    for file in ${topic_files[$topic]}; do
        echo "- **[$file]($file)**" >> "$CROSS_REF_FILE"
    done
    echo "" >> "$CROSS_REF_FILE"
done

# Generate sitemap with hierarchical structure
echo -e "\n${PURPLE}🗺️ Generating sitemap...${NC}"

# Root level files
echo "### Root Directory" >> "$SITEMAP_FILE"
find . -maxdepth 1 -name "*.md" -type f | sort | while read -r file; do
    relative_path=${file#./}
    size=$(wc -c < "$file")
    echo "- **[$relative_path]($relative_path)** ($size bytes) - ${file_descriptions[$relative_path]}" >> "$SITEMAP_FILE"
done
echo "" >> "$SITEMAP_FILE"

# Docs directory structure
echo "### Documentation Directory" >> "$SITEMAP_FILE"
find docs -name "*.md" -type f -not -path "docs/archive/*" | sort | while read -r file; do
    size=$(wc -c < "$file")
    depth=$(echo "$file" | tr -cd '/' | wc -c)
    indent=$(printf "%*s" $((depth * 2)) "")
    filename=$(basename "$file")
    dir=$(dirname "$file")
    
    echo "${indent}- **[$filename]($file)** ($size bytes)" >> "$SITEMAP_FILE"
done
echo "" >> "$SITEMAP_FILE"

# Generate quick navigation section
cat >> "$CROSS_REF_FILE" << EOF
## Quick Navigation

### By Size
$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" -exec wc -c {} + | \
  sort -rn | head -10 | while read size file; do
    echo "- **$size bytes** - [$file]($file)"
  done)

### Recently Modified
$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" -exec ls -lt {} + | \
  head -6 | tail -5 | while read -r line; do
    file=$(echo "$line" | awk '{print $NF}')
    date=$(echo "$line" | awk '{print $6, $7, $8}')
    echo "- **$date** - [$file]($file)"
  done)

EOF

# Generate related files suggestions
echo -e "\n${PURPLE}🤝 Generating related file suggestions...${NC}"

cat >> "$CROSS_REF_FILE" << EOF
## Suggested Reading Paths

### For New Users
1. [README.md](../README.md) - Project overview
2. [CLAUDE.md](../CLAUDE.md) - Essential configuration
3. [docs/deployment/getting-started.md](docs/deployment/getting-started.md) - Quick start
4. [docs/README.md](docs/README.md) - Documentation index

### For Developers
1. [docs/architecture/system-overview.md](docs/architecture/system-overview.md) - System architecture
2. [docs/architecture/engine-specifications.md](docs/architecture/engine-specifications.md) - Engine details
3. [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md) - API documentation
4. [docs/templates/engine-documentation-template.md](docs/templates/engine-documentation-template.md) - Documentation templates

### For Operations
1. [docs/deployment/production-deployment-guide.md](docs/deployment/production-deployment-guide.md) - Production deployment
2. [docs/deployment/troubleshooting.md](docs/deployment/troubleshooting.md) - Troubleshooting
3. [docs/performance/benchmarks.md](docs/performance/benchmarks.md) - Performance metrics
4. [docs/automation/doc-maintenance.sh](docs/automation/doc-maintenance.sh) - Maintenance automation

EOF

# Add footer
cat >> "$CROSS_REF_FILE" << EOF

---

**Generated by**: Nautilus Cross-Reference Generator  
**Command**: \`./docs/automation/generate-cross-references.sh\`  
**Last Updated**: $(date)

## Maintenance

This cross-reference file is automatically generated. To update:

\`\`\`bash
./docs/automation/generate-cross-references.sh
\`\`\`

## Validation

Validate all links with:

\`\`\`bash
./docs/automation/link-validator.sh
\`\`\`
EOF

# Add sitemap footer
cat >> "$SITEMAP_FILE" << EOF

---

**Generated by**: Nautilus Cross-Reference Generator  
**Last Updated**: $(date)  
**Total Files Indexed**: $(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | wc -l)

## File Statistics

$(echo "### Size Distribution")
$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" -exec wc -c {} + | \
  awk '$1 <= 5000 {small++} $1 > 5000 && $1 <= 10000 {medium++} $1 > 10000 && $1 <= 15000 {large++} $1 > 15000 {xl++} END {
    print "- Small (< 5KB): " (small ? small : 0)
    print "- Medium (5-10KB): " (medium ? medium : 0)
    print "- Large (10-15KB): " (large ? large : 0)
    print "- Extra Large (> 15KB): " (xl ? xl : 0)
  }')
EOF

# Summary
echo -e "\n${BLUE}📊 Cross-Reference Generation Complete${NC}"
echo "================================================"
echo -e "Files processed: ${GREEN}$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | wc -l)${NC}"
echo -e "Topics identified: ${GREEN}${#topic_files[@]}${NC}"
echo -e "Engine files: ${GREEN}${#engine_files[@]}${NC}"
echo -e "API files: ${GREEN}${#api_files[@]}${NC}"
echo -e "Architecture files: ${GREEN}${#architecture_files[@]}${NC}"

echo -e "\n${GREEN}✅ Generated Files:${NC}"
echo "  📋 Cross-references: $CROSS_REF_FILE"
echo "  🗺️  Sitemap: $SITEMAP_FILE"

echo -e "\n${YELLOW}💡 Next Steps:${NC}"
echo "  1. Review generated cross-references"
echo "  2. Run link validation: ./docs/automation/link-validator.sh"
echo "  3. Update documentation index if needed"

exit 0
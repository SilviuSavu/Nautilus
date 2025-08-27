#!/bin/bash
# Simplified Nautilus Cross-Reference Generator
# Creates cross-references between documentation files

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

CROSS_REF_FILE="docs/automation/cross-references.md"
SITEMAP_FILE="docs/automation/sitemap.md"

echo -e "${BLUE}🔗 Nautilus Cross-Reference Generator${NC}"
echo "=================================================="

# Initialize cross-reference file
cat > "$CROSS_REF_FILE" << EOF
# Documentation Cross-References

**Generated**: $(date)  
**Purpose**: Smart links between all documentation files

## File Categories

EOF

# Initialize sitemap
cat > "$SITEMAP_FILE" << EOF
# Documentation Sitemap

**Generated**: $(date)  
**Total Files**: $(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | wc -l)

## Project Structure

EOF

echo -e "\n${PURPLE}🔍 Categorizing documentation files...${NC}"

# Engine documentation
echo "### Engine Documentation" >> "$CROSS_REF_FILE"
find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | \
  grep -i "engine\|port.*8[0-9][0-9][0-9]" | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$(basename "$file")")
    echo "- **[$title]($file)** ($size bytes)" >> "$CROSS_REF_FILE"
done
echo "" >> "$CROSS_REF_FILE"

# API documentation
echo "### API Documentation" >> "$CROSS_REF_FILE"
find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | \
  grep -i "api\|endpoint\|rest\|websocket" | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$(basename "$file")")
    echo "- **[$title]($file)** ($size bytes)" >> "$CROSS_REF_FILE"
done
echo "" >> "$CROSS_REF_FILE"

# Architecture documentation
echo "### Architecture Documentation" >> "$CROSS_REF_FILE"
find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | \
  grep -i "architecture\|system\|design\|m4.*max" | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$(basename "$file")")
    echo "- **[$title]($file)** ($size bytes)" >> "$CROSS_REF_FILE"
done
echo "" >> "$CROSS_REF_FILE"

# Deployment documentation
echo "### Deployment Documentation" >> "$CROSS_REF_FILE"
find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | \
  grep -i "deploy\|docker\|getting.*started\|setup" | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$(basename "$file")")
    echo "- **[$title]($file)** ($size bytes)" >> "$CROSS_REF_FILE"
done
echo "" >> "$CROSS_REF_FILE"

# Generate topic-based quick navigation
echo -e "\n${PURPLE}📋 Generating topic navigation...${NC}"

cat >> "$CROSS_REF_FILE" << EOF
## Quick Navigation by Topic

### Getting Started
- **[README.md](../README.md)** - Project overview and quick start
- **[CLAUDE.md](../CLAUDE.md)** - Essential configuration for Claude Code
- **[Getting Started Guide](docs/deployment/getting-started.md)** - Comprehensive setup

### Architecture & Design  
- **[System Overview](docs/architecture/system-overview.md)** - High-level architecture
- **[Engine Specifications](docs/architecture/engine-specifications.md)** - All 12 engines
- **[M4 Max Optimization](docs/architecture/m4-max-optimization.md)** - Hardware acceleration
- **[MessageBus Architecture](docs/architecture/messagebus-architecture.md)** - Event system

### Performance & Optimization
- **[Performance Benchmarks](docs/performance/benchmarks.md)** - Validated metrics
- **[Hardware Acceleration](docs/architecture/m4-max-optimization.md)** - 50x+ improvements
- **[Engine Performance](docs/architecture/engine-specifications.md)** - Individual engine metrics

### API Reference
- **[API Reference](docs/api/API_REFERENCE.md)** - Complete REST API documentation
- **[WebSocket Endpoints](docs/api/WEBSOCKET_ENDPOINTS.md)** - Real-time streaming
- **[VPIN API](docs/api/VPIN_API_REFERENCE.md)** - Market microstructure endpoints

### Operations & Deployment
- **[Production Deployment](docs/deployment/production-deployment-guide.md)** - Production setup
- **[Docker Configuration](docs/deployment/docker-setup.md)** - Container management
- **[Troubleshooting](docs/deployment/troubleshooting.md)** - Common issues

### Documentation Management
- **[Documentation Standards](docs/standards/documentation-standards.md)** - Enterprise standards
- **[Templates](docs/templates/engine-documentation-template.md)** - Documentation templates
- **[Automation Scripts](docs/automation/)** - Maintenance tools

EOF

# Generate hierarchical sitemap
echo -e "\n${PURPLE}🗺️ Generating hierarchical sitemap...${NC}"

echo "### Root Level" >> "$SITEMAP_FILE"
find . -maxdepth 1 -name "*.md" -type f | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$(basename "$file")")
    echo "- **[$title]($file)** ($size bytes)" >> "$SITEMAP_FILE"
done
echo "" >> "$SITEMAP_FILE"

echo "### Documentation Directory Structure" >> "$SITEMAP_FILE"
find docs -name "*.md" -type f -not -path "docs/archive/*" | sort | while read -r file; do
    size=$(wc -c < "$file" 2>/dev/null || echo "0")
    depth=$(echo "$file" | tr -cd '/' | wc -c)
    indent=$(printf "%*s" $((depth * 2)) "")
    filename=$(basename "$file")
    title=$(head -1 "$file" 2>/dev/null | sed 's/^# *//' || echo "$filename")
    
    echo "${indent}- **[$title]($file)** ($size bytes)" >> "$SITEMAP_FILE"
done
echo "" >> "$SITEMAP_FILE"

# Add file statistics
cat >> "$SITEMAP_FILE" << EOF

## File Statistics

$(echo "### Size Distribution")
$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" -exec wc -c {} + | \
  awk '$1 <= 5000 {small++} $1 > 5000 && $1 <= 10000 {medium++} $1 > 10000 && $1 <= 15000 {large++} $1 > 15000 {xl++} END {
    print "- **Small (< 5KB)**: " (small ? small : 0) " files"
    print "- **Medium (5-10KB)**: " (medium ? medium : 0) " files"  
    print "- **Large (10-15KB)**: " (large ? large : 0) " files"
    print "- **Extra Large (> 15KB)**: " (xl ? xl : 0) " files"
  }')

### Recently Modified Files
$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" -exec ls -lt {} + | \
  head -6 | tail -5 | while read -r line; do
    file=$(echo "$line" | awk '{print $NF}')
    date=$(echo "$line" | awk '{print $6, $7, $8}')
    echo "- **$date**: [$file]($file)"
  done)

---

**Generated by**: Simplified Cross-Reference Generator  
**Last Updated**: $(date)  
**Command**: \`./docs/automation/simple-cross-references.sh\`
EOF

# Add footer to cross-references
cat >> "$CROSS_REF_FILE" << EOF

---

**Generated by**: Nautilus Cross-Reference Generator  
**Command**: \`./docs/automation/simple-cross-references.sh\`  
**Last Updated**: $(date)

## Maintenance Commands

\`\`\`bash
# Regenerate cross-references
./docs/automation/simple-cross-references.sh

# Validate all links
./docs/automation/link-validator.sh

# Check documentation health
./docs/automation/doc-maintenance.sh
\`\`\`
EOF

# Summary
total_files=$(find . -name "*.md" -type f -not -path "./docs/archive/*" -not -path "./node_modules/*" | wc -l)

echo -e "\n${BLUE}📊 Cross-Reference Generation Complete${NC}"
echo "================================================"
echo -e "Files processed: ${GREEN}$total_files${NC}"
echo -e "Cross-references: ${GREEN}$CROSS_REF_FILE${NC}"
echo -e "Sitemap: ${GREEN}$SITEMAP_FILE${NC}"

echo -e "\n${GREEN}✅ Smart cross-referencing system deployed!${NC}"
echo -e "View results: cat $CROSS_REF_FILE"

exit 0
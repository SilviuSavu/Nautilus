# Check Documentation Health Task

**Task ID**: check-doc-health  
**Agent**: doc-health  
**Category**: Quality Assurance  
**Priority**: High

## Overview

Performs a comprehensive health check on all project documentation, analyzing file sizes, link integrity, content structure, and compliance with enterprise standards. Generates detailed health reports with actionable recommendations.

## Task Description

This task executes a complete documentation analysis using the enterprise documentation automation system. It scans all markdown files, validates links, checks file sizes for Claude Code compatibility, and generates health scores with detailed metrics.

## Capabilities

### Health Analysis
- **File Size Validation**: Ensures Claude Code compatibility (<15KB optimal)
- **Link Integrity**: Validates all internal and external documentation links
- **Structure Analysis**: Checks heading hierarchy and document organization  
- **Standards Compliance**: Validates naming conventions and formatting
- **Cross-Reference Validation**: Ensures document relationships are maintained

### Reporting
- **Health Score**: 0-100% comprehensive quality metric
- **Detailed Breakdown**: File-by-file analysis with specific issues
- **Trend Tracking**: Compare with previous health checks
- **Actionable Recommendations**: Prioritized improvement suggestions
- **Executive Summary**: High-level overview for management

### Real-time Monitoring
- **Status Line Integration**: Updates Claude Code status line with health score
- **Alert System**: Proactive notifications for quality degradation
- **Automated Fixes**: Optional automatic resolution of common issues
- **Progress Tracking**: Monitor improvements over time

## Input Parameters

### Required
```yaml
project_path: .              # Path to project root (default: current directory)
```

### Optional
```yaml
output_format: markdown      # Output format: markdown, json, html
detailed: true               # Include detailed file-by-file analysis
fix_links: false            # Automatically fix broken internal links  
generate_sitemap: true      # Generate cross-reference sitemap
exclude_paths:              # Additional paths to exclude
  - custom-archive/
  - temp-docs/
include_external: true      # Check external links (may be slow)
cache_results: true         # Cache results for faster subsequent runs
```

## Execution Flow

### 1. Project Discovery
```bash
📚 Discovering documentation files...
✅ Found 1,406 markdown files
✅ Excluded 234 archive/node_modules files  
✅ Ready to analyze 1,172 active documentation files
```

### 2. Health Analysis
```bash
📊 Analyzing file sizes...
   - Optimal (< 10KB): 1,048 files
   - Acceptable (10-15KB): 98 files
   - Large (15-20KB): 22 files  
   - Critical (> 20KB): 4 files ❌

🔗 Validating links...
   - Internal links: 2,847 (2,845 valid, 2 broken)
   - External links: 456 (454 valid, 2 broken)
   
📋 Checking structure...
   - Proper headings: 1,168/1,172 files
   - Missing H1: 4 files ❌
```

### 3. Report Generation
```bash
📈 Generating health report...
   - Health Score: 94% 🎉 EXCELLENT
   - Detailed analysis: docs/automation/health-report.md
   - Sitemap: docs/automation/sitemap.md
   - Status line updated: 📚 94% 🎉
```

## Output Artifacts

### Health Report (`docs/automation/health-report.md`)
```markdown
# Documentation Health Report

**Generated**: August 26, 2025 01:00:00
**Health Score**: 94% 🎉 **EXCELLENT**

## Executive Summary
Your documentation maintains excellent quality with minimal issues requiring attention.

## Key Metrics
- Total Files: 1,172 
- Healthy Files: 1,102 (94%)
- Issues Found: 70 (6%)
- Broken Links: 4 (<1%)

## Critical Issues (Immediate Action Required)
- 4 files exceed 20KB (Claude Code compatibility)
- 2 broken internal links in API documentation

## Recommendations
1. Split 4 oversized files into modular sections
2. Fix broken links in docs/api/authentication.md
3. Add missing H1 headings to 4 files
```

### Sitemap (`docs/automation/sitemap.md`)
```markdown
# Documentation Sitemap

**Generated**: August 26, 2025
**Total Files**: 1,172

## Quick Navigation
- [API Documentation](docs/api/) - 45 files
- [Architecture](docs/architecture/) - 12 files  
- [Deployment](docs/deployment/) - 8 files
- [Integration](docs/integration/) - 23 files
```

### JSON Output (`docs/automation/health-data.json`)
```json
{
  "timestamp": "2025-08-26T01:00:00Z",
  "health_score": 94,
  "status": "excellent", 
  "summary": {
    "total_files": 1172,
    "healthy_files": 1102,
    "issues_found": 70,
    "broken_links": 4
  },
  "details": {
    "oversized_files": [...],
    "broken_links": [...],
    "recommendations": [...]
  }
}
```

## Usage Examples

### Basic Health Check
```bash
# Run standard health check
bmad run check-doc-health

# Quick check without external links
bmad run check-doc-health include_external=false

# Generate detailed report
bmad run check-doc-health detailed=true output_format=html
```

### Advanced Usage
```bash
# Check specific directory
bmad run check-doc-health project_path=./docs

# Auto-fix common issues
bmad run check-doc-health fix_links=true

# Custom exclusions
bmad run check-doc-health exclude_paths="['legacy/', 'temp/']"
```

### Integration with Other Tasks
```bash
# Chain with link validation
bmad run check-doc-health && bmad run validate-doc-links

# Schedule regular checks
bmad schedule check-doc-health --daily
```

## Error Handling

### Common Issues
- **Permission denied**: Ensure read access to all documentation directories
- **Large projects**: Use `cache_results=true` for faster subsequent runs
- **Network timeouts**: Set `include_external=false` for offline usage
- **Disk space**: Reports can be large; ensure adequate space

### Recovery Actions
```bash
# Reset cache if corrupted
bmad run check-doc-health --reset-cache

# Skip problematic files
bmad run check-doc-health exclude_paths="['problematic/']"

# Generate minimal report
bmad run check-doc-health detailed=false include_external=false
```

## Performance Optimization

### For Large Projects (>1000 files)
- Enable caching: `cache_results=true`
- Exclude unnecessary paths: `exclude_paths`
- Skip external links: `include_external=false`
- Use incremental checks: `--incremental`

### For CI/CD Integration  
- Use JSON output: `output_format=json`
- Set fail thresholds: `--fail-below=80`
- Generate badges: `--generate-badge`
- Cache between runs: `--cache-dir=.ci-cache`

## Success Metrics

- **Health Score >90%**: Excellent documentation quality
- **<5% Broken Links**: Strong link integrity
- **100% Claude Code Compatible**: All files <15KB
- **Consistent Structure**: Proper heading hierarchy
- **Automated Monitoring**: Real-time status tracking

---

**Task Version**: 1.0.0  
**Agent Compatibility**: doc-health >=1.0.0  
**Dependencies**: bash, curl, find, grep, wc  
**Estimated Runtime**: 30s - 5min (depending on project size)
# Validate Documentation Links Task

**Task ID**: validate-doc-links  
**Agent**: doc-health  
**Category**: Quality Assurance  
**Priority**: Medium

## Overview

Performs comprehensive link validation across all project documentation, checking internal file references, external URLs, and anchor links. Generates detailed reports of broken links with suggested fixes and automatic repair options.

## Task Description

This task systematically validates all markdown links in the project documentation. It distinguishes between internal file links, external HTTP/HTTPS URLs, and anchor links, providing specific validation and repair strategies for each type.

## Capabilities

### Link Types Supported
- **Internal File Links**: `[text](../path/to/file.md)`, `[text](/absolute/path.md)`
- **External URLs**: `[text](https://example.com)`, `[text](http://api.service.com)`
- **Anchor Links**: `[text](#section-heading)`, `[text](file.md#anchor)`
- **Email Links**: `[text](mailto:email@example.com)`
- **Protocol Links**: `[text](ftp://server.com)`, `[text](ssh://server.com)`

### Validation Features
- **Real-time Checking**: Live validation of external URLs with timeout handling
- **Path Resolution**: Intelligent relative and absolute path resolution
- **Anchor Validation**: Verification that anchor targets exist in referenced files
- **Redirect Following**: Follows HTTP redirects to determine final link status
- **Batch Processing**: Efficient validation of large link collections
- **Cache Support**: Caches external URL results to avoid redundant requests

### Repair Capabilities
- **Automatic Fixes**: Auto-correct common path and casing issues
- **Interactive Repair**: Guided fix process with multiple options
- **Batch Updates**: Apply fixes across multiple files simultaneously  
- **Backup Creation**: Safe backup before making changes
- **Rollback Support**: Undo changes if issues arise

## Input Parameters

### Required
```yaml
project_path: .              # Path to project root
```

### Optional  
```yaml
external_links: true         # Check external HTTP/HTTPS URLs
anchor_links: true           # Validate anchor references
timeout: 10                  # Timeout for external URLs (seconds)
max_redirects: 3             # Maximum redirects to follow
concurrent_checks: 5         # Parallel external URL checks
output_format: markdown      # Output: markdown, json, csv, html
auto_fix: false             # Automatically fix broken internal links
interactive_fix: false      # Interactive repair mode
create_backup: true         # Backup files before fixes
exclude_domains:            # Skip specific external domains
  - "example.com"
  - "localhost"
include_patterns:           # Only check links matching patterns
  - "docs/**"
  - "*.md"
exclude_patterns:           # Skip links matching patterns
  - "node_modules/**"
  - ".git/**"
cache_external: true        # Cache external URL results
cache_duration: 3600        # Cache duration in seconds
```

## Execution Flow

### 1. Link Discovery
```bash
🔍 Discovering links in documentation...
   📄 Scanning 1,172 markdown files...
   🔗 Found 3,247 total links
   
   Link Types:
   - Internal file links: 2,134 (65.7%)
   - External URLs: 891 (27.4%)  
   - Anchor links: 167 (5.1%)
   - Email links: 42 (1.3%)
   - Other protocols: 13 (0.4%)
```

### 2. Internal Link Validation  
```bash
📁 Validating internal file links...
   ✅ Valid links: 2,131/2,134 (99.9%)
   ❌ Broken links: 3 (0.1%)
   
   Broken Internal Links:
   - docs/api/auth.md:42 → ../config/setup.md (File not found)
   - README.md:15 → docs/quickstart.md (File not found) 
   - docs/deploy.md:67 → ./scripts/deploy.sh (File not found)
```

### 3. External URL Validation
```bash
🌐 Validating external URLs...
   📡 Checking 891 external URLs...
   ⏱️  Timeout: 10s per URL, 5 concurrent
   
   ✅ Valid URLs: 887/891 (99.6%)
   ❌ Broken URLs: 4 (0.4%)
   
   Broken External Links:
   - docs/integration/api.md:23 → https://old-api.service.com (404 Not Found)
   - docs/references.md:156 → https://broken-link.com (Connection timeout)
```

### 4. Anchor Link Validation
```bash
⚓ Validating anchor links...
   📋 Checking 167 anchor references...
   
   ✅ Valid anchors: 164/167 (98.2%)
   ❌ Broken anchors: 3 (1.8%)
   
   Broken Anchor Links:
   - docs/api.md:78 → #authentication (Section not found)
   - README.md:34 → docs/setup.md#configuration (Anchor missing)
```

## Output Artifacts

### Link Validation Report (`docs/automation/link-report.md`)
```markdown
# Documentation Link Validation Report

**Generated**: August 26, 2025 01:15:00
**Total Links**: 3,247
**Valid Links**: 3,182 (98.0%)  
**Broken Links**: 65 (2.0%)

## Summary by Type
| Link Type | Total | Valid | Broken | Success Rate |
|-----------|-------|-------|--------|--------------|
| Internal  | 2,134 | 2,131 | 3      | 99.9%        |
| External  | 891   | 887   | 4      | 99.6%        |
| Anchors   | 167   | 164   | 3      | 98.2%        |
| Other     | 55    | 55    | 0      | 100%         |

## Broken Links Detail

### Internal File Links (3 broken)
1. **docs/api/auth.md:42**
   - Link: `../config/setup.md` 
   - Error: File not found
   - Suggested Fix: `../configuration/setup.md` (file exists)
   
2. **README.md:15**
   - Link: `docs/quickstart.md`
   - Error: File not found  
   - Suggested Fix: `docs/getting-started/quickstart.md` (file exists)

### External URLs (4 broken)
1. **docs/integration/api.md:23**
   - Link: `https://old-api.service.com`
   - Error: 404 Not Found
   - Suggested Fix: `https://new-api.service.com` (redirects found)

## Recommended Actions
1. Fix 3 broken internal links (auto-fixable)
2. Update 4 broken external URLs
3. Add missing anchor sections
4. Consider link monitoring for external URLs
```

### JSON Report (`docs/automation/link-data.json`)
```json
{
  "timestamp": "2025-08-26T01:15:00Z",
  "summary": {
    "total_links": 3247,
    "valid_links": 3182,
    "broken_links": 65,
    "success_rate": 98.0
  },
  "by_type": {
    "internal": {"total": 2134, "valid": 2131, "broken": 3},
    "external": {"total": 891, "valid": 887, "broken": 4},
    "anchors": {"total": 167, "valid": 164, "broken": 3}
  },
  "broken_links": [
    {
      "file": "docs/api/auth.md",
      "line": 42,
      "link": "../config/setup.md",
      "type": "internal",
      "error": "File not found",
      "suggested_fix": "../configuration/setup.md"
    }
  ]
}
```

## Usage Examples

### Basic Link Validation
```bash
# Validate all links
bmad run validate-doc-links

# Internal links only (fast)
bmad run validate-doc-links external_links=false

# External links with custom timeout
bmad run validate-doc-links timeout=30
```

### Advanced Usage
```bash
# Auto-fix broken internal links
bmad run validate-doc-links auto_fix=true

# Interactive repair mode
bmad run validate-doc-links interactive_fix=true

# Skip specific domains
bmad run validate-doc-links exclude_domains="['old-domain.com', 'deprecated.com']"

# High-performance mode
bmad run validate-doc-links concurrent_checks=10 cache_external=true
```

### Integration with CI/CD
```bash
# Generate JSON report for CI
bmad run validate-doc-links output_format=json

# Fail build if >5% broken links
bmad run validate-doc-links --fail-threshold=5

# Cache results between CI runs  
bmad run validate-doc-links cache_external=true cache_duration=7200
```

## Interactive Fix Mode

When `interactive_fix=true`, the task provides guided repair:

```bash
🔧 Interactive Link Repair Mode

Broken link found: docs/api/auth.md:42 → ../config/setup.md

Available actions:
1. 🎯 Use suggested fix: ../configuration/setup.md
2. 📝 Manually enter correct path
3. 🗑️  Remove the broken link
4. ⏭️  Skip this link
5. 🚫 Cancel repair process

Your choice [1-5]: 1

✅ Fixed: ../config/setup.md → ../configuration/setup.md
📝 Backup created: docs/api/auth.md.backup

Continue with next broken link? [Y/n]: Y
```

## Automation Features

### Scheduled Validation
```bash
# Daily link validation
bmad schedule validate-doc-links --daily

# Weekly comprehensive check
bmad schedule validate-doc-links --weekly external_links=true timeout=60

# Before deployments
bmad hook pre-deploy validate-doc-links --fail-threshold=0
```

### Integration with Health Checks
```bash
# Combined health and link check
bmad run check-doc-health && bmad run validate-doc-links

# Update health score after link fixes
bmad run validate-doc-links auto_fix=true && bmad run check-doc-health
```

## Performance Optimization

### Large Projects
- Increase `concurrent_checks` for external URLs
- Enable `cache_external` for repeated runs
- Use `include_patterns` to limit scope
- Process in batches with `--batch-size`

### External URL Handling
- Set appropriate `timeout` values
- Limit `max_redirects` for security
- Use `exclude_domains` for known issues
- Cache results with `cache_duration`

## Error Recovery

### Common Issues
- **Network timeouts**: Increase `timeout` or exclude problematic domains
- **Permission errors**: Ensure read access to all documentation
- **Cache corruption**: Clear cache with `--clear-cache`
- **Large repairs**: Use `create_backup=true` for safety

### Rollback Procedures
```bash
# Restore from backups
bmad run validate-doc-links --restore-backups

# Selective rollback
bmad run validate-doc-links --rollback-file=docs/api/auth.md
```

---

**Task Version**: 1.0.0  
**Agent Compatibility**: doc-health >=1.0.0  
**Dependencies**: bash, curl, find, grep  
**Estimated Runtime**: 1-15min (depending on external URLs)
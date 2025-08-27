# Enforce Documentation Standards Task

**Task ID**: enforce-doc-standards  
**Agent**: doc-health  
**Category**: Standards & Compliance  
**Priority**: High

## Overview

Applies enterprise-grade documentation standards across the entire project, ensuring consistency, professional quality, and Claude Code compatibility. Automatically fixes common issues and provides guidance for complex standardization requirements.

## Task Description

This task systematically reviews all project documentation against enterprise standards, automatically correcting formatting issues, enforcing naming conventions, ensuring proper structure, and maintaining Claude Code compatibility. It provides both automated fixes and interactive guidance for complex standardization needs.

## Capabilities

### Standards Enforcement
- **File Size Management**: Ensures Claude Code compatibility (<15KB optimal)
- **Naming Conventions**: Standardizes file and directory naming patterns
- **Content Structure**: Enforces proper heading hierarchy and organization
- **Formatting Standards**: Applies consistent markdown formatting rules
- **Template Compliance**: Ensures documents follow established templates

### Automatic Corrections
- **Heading Standardization**: Fixes heading levels and hierarchy
- **Link Format Correction**: Standardizes internal and external link formats
- **Code Block Enhancement**: Adds language specifications and proper formatting
- **Metadata Addition**: Adds missing front matter and document metadata
- **Cross-Reference Updates**: Maintains consistent cross-reference formats

### Quality Improvements
- **Content Organization**: Restructures documents for better readability
- **Professional Formatting**: Applies enterprise-grade presentation standards
- **Accessibility Enhancements**: Improves document accessibility and navigation
- **SEO Optimization**: Enhances discoverability through better structure

## Standards Enforced

### File Organization Standards
```yaml
Naming Conventions:
  files: "lowercase-with-hyphens.md"         # Preferred
  directories: "lowercase-with-hyphens/"     # Consistent structure
  legacy_support: "UPPERCASE_UNDERSCORES.md" # Acceptable

Size Requirements:
  optimal: "<10KB"      # Easy to read and edit
  acceptable: "10-15KB" # Manageable size  
  warning: "15-20KB"    # Consider splitting
  critical: ">20KB"     # BREAKS Claude Code compatibility

Directory Structure:
  docs/
    ├── README.md              # Master index
    ├── standards/             # Documentation standards
    ├── templates/            # Document templates
    ├── automation/           # Maintenance scripts
    ├── architecture/         # System design documents
    ├── api/                  # API documentation
    ├── deployment/           # Deployment guides
    └── archive/              # Historical content
```

### Content Structure Standards
```yaml
Document Template:
  - H1 Title (single per document)
  - Brief description in bold
  - H2 Overview section (required)
  - Logical H2/H3 hierarchy
  - H2 References section (if applicable)
  - Footer with status and dates

Markdown Standards:
  - Code blocks with language specification
  - Consistent bullet points (- preferred)
  - Proper link formatting [text](url)
  - Status indicators with emojis
  - Professional tone and grammar
```

### Professional Formatting
```yaml
Status Indicators:
  ✅ OPERATIONAL: Working perfectly
  ⚠️ WARNING: Attention needed but functional
  ❌ FAILED: Not working, needs immediate attention
  🔧 MAINTENANCE: Under active development
  📋 PLANNED: Future implementation
  🚨 CRITICAL: Mission-critical importance

Code Documentation:
  - Always specify language in code blocks
  - Include comments explaining commands
  - Use consistent indentation
  - Add examples where helpful

Professional Elements:
  - Executive summaries for complex topics
  - Performance metrics in tabular format
  - Visual diagrams using Mermaid when beneficial
  - Clear call-to-action sections
```

## Input Parameters

### Required
```yaml
project_path: .              # Path to project root
```

### Optional
```yaml
standards_profile: enterprise    # Standards level: basic, professional, enterprise
auto_fix: true                  # Automatically fix issues where possible
interactive_mode: false         # Interactive guidance for complex issues
create_backups: true            # Backup files before modifications
apply_templates: true           # Apply standard templates to documents
enforce_naming: true            # Fix file and directory naming
restructure_content: false     # Reorganize content structure (careful!)
add_metadata: true              # Add missing document metadata
update_crossrefs: true          # Update cross-references to new standards
generate_missing: true          # Generate missing standard documents
exclude_patterns:               # Skip files matching patterns
  - "node_modules/**"
  - ".git/**"
  - "**/*.backup"
include_patterns:               # Only process files matching patterns
  - "**/*.md"
  - "**/*.markdown"
custom_standards:               # Override default standards
  file_size_limit: 15000
  heading_style: "atx"          # ATX (#) vs Setext (===)
  bullet_style: "-"            # - vs * vs +
  code_fence: "```"            # ``` vs ~~~
```

## Execution Flow

### 1. Standards Analysis
```bash
📏 Analyzing current documentation standards...
   📊 Scanning 1,172 files for compliance
   📋 Checking against enterprise standards profile
   
   Current Compliance:
   - File naming: 847/1,172 (72.3%) compliant
   - Size limits: 1,098/1,172 (93.7%) compliant  
   - Structure: 934/1,172 (79.7%) compliant
   - Formatting: 723/1,172 (61.7%) compliant
   
   Overall Standards Score: 76.9% ⚠️ NEEDS IMPROVEMENT
```

### 2. Automatic Corrections
```bash
🔧 Applying automatic corrections...
   
   ✅ Fixed file naming issues:
   - api_reference.md → api-reference.md
   - User Guide.md → user-guide.md  
   - deployment_HOWTO.md → deployment-howto.md
   
   ✅ Corrected heading hierarchy:
   - Fixed 34 documents with multiple H1 headings
   - Adjusted 89 documents with improper nesting
   - Added missing H1 headings to 12 documents
   
   ✅ Enhanced code blocks:
   - Added language specs to 156 code blocks
   - Fixed indentation in 67 code examples
   - Added explanatory comments to 89 bash scripts
   
   ✅ Updated cross-references:
   - Fixed 23 broken internal links due to renames
   - Updated 67 references to follow new standards
```

### 3. Template Application
```bash
📋 Applying standard templates...
   
   ✅ API Documentation Template:
   - Applied to 23 API documentation files
   - Added standard sections: Overview, Endpoints, Examples
   - Included performance metrics tables
   
   ✅ Architecture Template:
   - Applied to 12 architecture documents
   - Added Mermaid diagrams where appropriate
   - Included standard technical specifications
   
   ✅ Deployment Template:
   - Applied to 8 deployment guides  
   - Added prerequisite checklists
   - Included troubleshooting sections
```

### 4. Quality Enhancement
```bash
🎯 Enhancing document quality...
   
   ✅ Added professional elements:
   - Status indicators to 234 documents
   - Performance metrics tables to 45 technical docs
   - Executive summaries to 12 complex documents
   
   ✅ Improved accessibility:
   - Added alt text to 67 images
   - Enhanced table headers for screen readers  
   - Improved link descriptions for context
   
   ✅ SEO optimization:
   - Enhanced headings for better structure
   - Added relevant keywords naturally
   - Improved cross-reference anchor text
```

## Output Artifacts

### Standards Compliance Report (`docs/automation/standards-report.md`)
```markdown
# Documentation Standards Compliance Report

**Generated**: August 26, 2025 01:45:00
**Standards Profile**: Enterprise
**Overall Compliance**: 92.4% ✅ **EXCELLENT** (improved from 76.9%)

## Summary of Changes

### Automatic Corrections Applied
- **File Naming**: Fixed 89 files (100% compliance achieved)
- **Heading Structure**: Corrected 135 documents  
- **Code Block Standards**: Enhanced 312 code examples
- **Cross-References**: Updated 90 internal links

### Template Applications
- **API Documentation**: 23 files standardized
- **Architecture Documents**: 12 files enhanced
- **Deployment Guides**: 8 files restructured

### Quality Improvements
- **Status Indicators**: Added to 234 documents
- **Performance Tables**: Added to 45 technical documents
- **Accessibility**: Improved 67 documents

## Current Compliance Metrics

| Standard Category | Before | After | Improvement |
|-------------------|--------|-------|-------------|
| File Naming       | 72.3%  | 100%  | +27.7%      |
| Size Limits       | 93.7%  | 94.1% | +0.4%       |
| Structure         | 79.7%  | 96.8% | +17.1%      |
| Formatting        | 61.7%  | 89.2% | +27.5%      |

## Remaining Issues

### Manual Review Required (8 files)
1. **docs/legacy/old-api.md** - Complex restructuring needed
2. **docs/migration/v1-to-v2.md** - Content organization review
3. **docs/advanced/custom-integrations.md** - Technical accuracy check

### Size Warnings (4 files) 
1. **docs/api/complete-reference.md** (18.2 KB) - Consider splitting
2. **docs/architecture/system-design.md** (16.8 KB) - Review sections
3. **docs/deployment/enterprise-setup.md** (17.1 KB) - Modularize steps
4. **docs/integration/complex-scenarios.md** (16.4 KB) - Split use cases

## Recommendations
1. ✅ Review 8 documents requiring manual attention
2. ⚠️ Consider splitting 4 oversized documents  
3. 📋 Schedule monthly standards compliance checks
4. 🎯 Maintain >90% compliance score
```

### Standards Configuration (`docs/standards/applied-standards.yaml`)
```yaml
# Applied Documentation Standards
applied_date: "2025-08-26"
standards_profile: enterprise
version: 1.0.0

file_standards:
  naming_convention: lowercase-with-hyphens
  size_limit_bytes: 15000
  size_warning_bytes: 10000
  
content_standards:
  heading_style: atx
  bullet_style: "-"
  code_fence_style: "```"
  require_language_spec: true
  
structure_standards:
  require_h1: true
  single_h1_only: true
  require_overview: true
  max_heading_depth: 4
  
formatting_standards:
  status_indicators: true
  performance_tables: true
  code_comments: true
  
compliance_tracking:
  last_check: "2025-08-26T01:45:00Z"
  next_scheduled: "2025-09-26T01:45:00Z"
  compliance_score: 92.4
  target_score: 90.0
```

## Usage Examples

### Basic Standards Enforcement
```bash
# Apply all enterprise standards
bmad run enforce-doc-standards

# Basic standards only
bmad run enforce-doc-standards standards_profile=basic

# Preview changes without applying
bmad run enforce-doc-standards auto_fix=false
```

### Advanced Usage
```bash
# Interactive mode for complex issues
bmad run enforce-doc-standards interactive_mode=true

# Custom standards configuration
bmad run enforce-doc-standards custom_standards="{'file_size_limit': 12000}"

# Focus on specific areas
bmad run enforce-doc-standards include_patterns="['docs/api/**']"
```

### Integration Examples
```bash
# After major documentation updates
bmad run enforce-doc-standards create_backups=true

# Scheduled compliance checks
bmad schedule enforce-doc-standards --monthly

# CI/CD integration
bmad run enforce-doc-standards --fail-below=85 --quiet
```

## Interactive Mode Features

When `interactive_mode=true`, provides guided assistance:

```bash
🎯 Interactive Standards Enforcement

Complex issue detected: docs/api/legacy.md

Current issues:
- Multiple H1 headings (3 found)
- No overview section
- Inconsistent code block formatting
- Size: 16.8 KB (approaching limit)

Suggested actions:
1. 📋 Restructure as multiple documents
2. 🔧 Apply API documentation template  
3. ✂️ Split into focused sections
4. 📝 Manual review and cleanup

Your choice [1-4]: 1

🔄 Restructuring into focused documents:
- docs/api/legacy-authentication.md
- docs/api/legacy-endpoints.md  
- docs/api/legacy-migration.md

Proceed with restructure? [Y/n]: Y

✅ Successfully restructured legacy.md into 3 focused documents
📋 Updated cross-references in 12 related files
🎯 All documents now meet enterprise standards
```

## Quality Assurance Features

### Backup and Recovery
- Automatic backup creation before changes
- Rollback capability for any modifications
- Change tracking for audit purposes
- Safe operation with version control

### Validation Checks
- Pre-enforcement standards analysis
- Post-enforcement compliance verification
- Cross-reference integrity validation
- Template application verification

### Reporting and Tracking
- Detailed compliance reports
- Before/after comparison metrics
- Progress tracking over time
- Standards evolution documentation

## Advanced Configuration

### Custom Standards Profiles
```yaml
# custom-standards.yaml
profiles:
  startup:
    file_size_limit: 20000
    require_overview: false
    
  enterprise:
    file_size_limit: 15000
    require_overview: true
    require_performance_tables: true
    
  government:
    accessibility_required: true
    section_508_compliance: true
    audit_trail_required: true
```

### Industry-Specific Standards
- **Technical Documentation**: API-first, code-heavy standards
- **User Documentation**: Accessibility and readability focused
- **Regulatory Compliance**: Audit trails and version control
- **Open Source**: Community-friendly and contribution-ready

---

**Task Version**: 1.0.0  
**Agent Compatibility**: doc-health >=1.0.0  
**Dependencies**: bash, find, grep, sed, backup utilities  
**Estimated Runtime**: 2-10min (depending on project size and fixes needed)
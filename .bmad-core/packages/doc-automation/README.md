# BMAD Documentation Automation Package

**Version**: 1.0.0  
**Category**: Documentation & Quality Assurance  
**Agent**: Dr. DocHealth 📚  

## Overview

**The BMAD Documentation Automation Package is an enterprise-grade system that transforms documentation management from a manual chore into an intelligent, automated workflow.** This comprehensive solution provides real-time health monitoring, automatic quality fixes, intelligent cross-referencing, and professional templates - all integrated seamlessly with Claude Code and the BMAD framework.

## 🚀 Key Features

### Real-Time Health Monitoring
- **Live Status Line**: Documentation health displayed in Claude Code status bar
- **Health Scoring**: 0-100% scoring system with 4 alert levels (Excellent, Good, Warning, Critical)
- **File-by-File Analysis**: Individual file health assessment with size, structure, and link validation
- **Automatic Updates**: Health scores update automatically when files are modified

### Intelligent Automation
- **Post-Edit Hooks**: Automatic processing when files are edited or created
- **Auto-Fixes**: Automatic correction of common issues (missing headings, formatting, etc.)
- **Link Validation**: Real-time validation of internal and external links
- **Background Monitoring**: Daemon service for continuous health monitoring

### Professional Templates
- **API Documentation**: Complete REST API documentation template
- **Architecture Documents**: System architecture and design templates  
- **Deployment Guides**: Comprehensive deployment and setup templates
- **Troubleshooting Guides**: Support and troubleshooting documentation templates

### Enterprise Standards
- **Size Management**: Claude Code compatibility (<15KB optimal, warnings for larger files)
- **Structure Enforcement**: Proper heading hierarchy and organization
- **Link Integrity**: Comprehensive link validation and repair
- **Cross-References**: Intelligent document relationship mapping

## 📦 Package Contents

### Core Components
```
doc-automation/
├── manifest.yaml                 # Package configuration
├── install.sh                   # One-click installer
└── README.md                    # This file

├── agents/                      # BMAD Agents
│   └── doc-health.md           # Dr. DocHealth agent

├── tasks/                       # BMAD Tasks
│   ├── check-doc-health.md     # Health monitoring task
│   ├── validate-doc-links.md   # Link validation task
│   ├── generate-doc-sitemap.md # Cross-reference generation
│   └── enforce-doc-standards.md # Standards enforcement

├── scripts/                     # Automation Scripts
│   ├── bmad-doc-maintenance.sh # Health monitoring script
│   ├── bmad-link-validator.sh  # Link validation script
│   └── bmad-cross-references.sh # Cross-reference generator

├── templates/                   # Documentation Templates
│   ├── api-documentation.md    # API documentation template
│   ├── architecture-document.md # Architecture template
│   ├── deployment-guide.md     # Deployment guide template
│   └── troubleshooting-guide.md # Troubleshooting template

└── hooks/                       # Integration Hooks
    ├── doc-health-status-line.sh # Status line integration
    ├── post-edit-hook.sh       # Post-edit automation
    └── install-hooks.sh         # Hook installer
```

## 🔧 Installation

### One-Click Installation
```bash
# Download and install in one command
curl -fsSL https://bmad.dev/install/doc-automation | bash

# Or clone and install locally
git clone https://github.com/bmad-framework/bmad-packages.git
cd bmad-packages/packages/doc-automation
chmod +x install.sh
./install.sh
```

### Manual Installation Steps
```bash
# 1. Install to BMAD packages directory
mkdir -p ~/.bmad/packages/doc-automation
cp -r * ~/.bmad/packages/doc-automation/

# 2. Install hooks and automation
~/.bmad/packages/doc-automation/hooks/install-hooks.sh install

# 3. Test installation
bmad-doc-automation health
```

### System Requirements
- **Operating System**: macOS 10.15+, Ubuntu 20.04+, CentOS 8+
- **Required**: bash, git, curl
- **Recommended**: fswatch (macOS) or inotify-tools (Linux), python3, jq
- **Optional**: Claude Code (for status line integration)

## 🎯 Quick Start

### 1. Check Documentation Health
```bash
# Quick health check
bmad-doc-automation health

# Or using BMAD framework
bmad run check-doc-health
```

### 2. Real-Time Status Monitoring
```bash
# Show current status with colors
bmad-doc-automation status --colored

# Install automatic status line updates
bmad-doc-automation hooks install
```

### 3. Validate Links
```bash
# Validate all documentation links
bmad-doc-automation validate-links

# Validate with auto-repair
bmad run validate-doc-links auto_fix=true
```

### 4. Generate Cross-References
```bash
# Generate documentation sitemap and cross-references
bmad-doc-automation generate-sitemap

# Generate with topic clustering
bmad run generate-doc-sitemap generate_topics=true
```

### 5. Apply Professional Templates
```bash
# List available templates
bmad-doc-automation templates list

# Apply API documentation template
bmad apply template api-documentation target=docs/api/new-service.md

# Apply architecture template
bmad apply template architecture-document target=docs/architecture/system.md
```

## 📊 Health Scoring System

The package uses a comprehensive 0-100% health scoring system:

### Scoring Criteria
- **File Size**: Penalties for files >10KB (Claude Code compatibility)
- **Structure**: Missing H1 headings, multiple H1s, poor hierarchy
- **Links**: Broken internal links, invalid external links
- **Standards**: Inconsistent formatting, naming conventions

### Health Levels
| Score | Status | Icon | Description |
|-------|---------|------|-------------|
| 95-100% | 🎉 **Excellent** | Green | All files optimal, no issues |
| 85-94% | ✅ **Good** | Green | Minor issues, mostly healthy |
| 70-84% | ⚠️ **Attention Needed** | Yellow | Several issues need addressing |
| 0-69% | ❌ **Critical** | Red | Major problems require immediate action |

### Example Health Report
```
📊 Documentation Health Analysis
================================
Overall Health Score: 94% ✅ GOOD
Total Files: 47
Issues Found: 3 minor

File Breakdown:
✅ Excellent (< 10KB): 41 files (87%)
⚠️  Large (10-15KB): 4 files (9%)
❌ Oversized (> 15KB): 2 files (4%)

Recommended Actions:
• Split 2 oversized files for Claude Code compatibility
• Fix 1 broken internal link
• Add H1 heading to 2 files
```

## 🤖 Dr. DocHealth Agent

The package includes "Dr. DocHealth", an intelligent BMAD agent for interactive documentation management.

### Agent Capabilities
- **Health Analysis**: Deep analysis of documentation health with actionable recommendations
- **Interactive Fixes**: Guided fixing of critical issues with user interaction
- **Quality Assessment**: Comprehensive quality evaluation with improvement suggestions
- **Standards Enforcement**: Automatic application of enterprise documentation standards

### Using the Agent
```bash
# Interactive health analysis
bmad agent doc-health analyze

# Fix critical issues with guidance
bmad agent doc-health fix-critical

# Optimize document structure
bmad agent doc-health optimize-structure

# Get improvement recommendations
bmad agent doc-health recommend
```

### Agent Personality
Dr. DocHealth is designed as a friendly, knowledgeable documentation specialist who:
- Provides clear, actionable advice
- Explains the reasoning behind recommendations
- Offers multiple solution approaches
- Maintains a professional but approachable tone

## 📋 BMAD Tasks Reference

### check-doc-health
**Purpose**: Comprehensive documentation health monitoring  
**Features**: File size analysis, structure validation, health scoring  
**Usage**: `bmad run check-doc-health detailed=true`

### validate-doc-links
**Purpose**: Link validation with automatic repair capabilities  
**Features**: Internal/external/anchor link checking, parallel processing, caching  
**Usage**: `bmad run validate-doc-links external_only=true`

### generate-doc-sitemap
**Purpose**: Intelligent cross-reference and sitemap generation  
**Features**: Topic extraction, hierarchical organization, multiple output formats  
**Usage**: `bmad run generate-doc-sitemap output_formats="['markdown','json']"`

### enforce-doc-standards
**Purpose**: Enterprise documentation standards enforcement  
**Features**: Automatic fixes, template application, naming conventions  
**Usage**: `bmad run enforce-doc-standards auto_fix=true`

## 📝 Templates Guide

### API Documentation Template
**File**: `templates/api-documentation.md`  
**Purpose**: Complete REST API documentation with examples, authentication, error handling  
**Sections**: Overview, authentication, endpoints, SDKs, testing, troubleshooting

**Usage**:
```bash
bmad apply template api-documentation target=docs/api/user-service.md
```

### Architecture Document Template
**File**: `templates/architecture-document.md`  
**Purpose**: System architecture and design documentation  
**Sections**: High-level architecture, components, data flow, security, scalability

**Usage**:
```bash
bmad apply template architecture-document target=docs/architecture/microservices.md
```

### Deployment Guide Template
**File**: `templates/deployment-guide.md`  
**Purpose**: Comprehensive deployment and operations guide  
**Sections**: Prerequisites, installation, configuration, monitoring, troubleshooting

**Usage**:
```bash
bmad apply template deployment-guide target=docs/deployment/production.md
```

### Troubleshooting Guide Template
**File**: `templates/troubleshooting-guide.md`  
**Purpose**: Support and problem resolution documentation  
**Sections**: Common issues, diagnostic steps, solutions, escalation procedures

**Usage**:
```bash
bmad apply template troubleshooting-guide target=docs/support/api-issues.md
```

## ⚙️ Configuration

### Environment Variables
```bash
# Core BMAD settings
export BMAD_ENABLED=true                    # Enable BMAD integration
export BMAD_AUTO_FIX=true                   # Enable automatic fixes
export BMAD_VALIDATE_ON_EDIT=true           # Validate links on file edit

# Health monitoring settings
export DOC_HEALTH_UPDATE_INTERVAL=300       # Update interval (seconds)
export DOC_HEALTH_MAX_FILE_SIZE=15000       # Max file size (bytes)
export DOC_HEALTH_WARNING_SIZE=10000        # Warning threshold (bytes)

# Link validation settings
export VALIDATE_EXTERNAL=true               # Validate external URLs
export VALIDATE_INTERNAL=true               # Validate internal links
export VALIDATE_ANCHORS=true                # Validate anchor links
export LINK_TIMEOUT_SECONDS=10              # URL validation timeout

# Debug and logging
export BMAD_DEBUG=true                      # Enable debug output
export BMAD_QUIET=false                     # Suppress non-essential output
```

### Claude Code Integration
The package automatically integrates with Claude Code for:
- **Status Line Updates**: Real-time health score in status bar
- **Post-Edit Hooks**: Automatic processing after file edits
- **Settings Configuration**: Automatic status line component registration

**Manual Claude Code Configuration**:
```json
{
  "statusLine": {
    "components": [
      {
        "name": "bmad-doc-health",
        "command": "~/.bmad/packages/doc-automation/hooks/doc-health-status-line.sh --status-short",
        "interval": 30,
        "format": "text"
      }
    ]
  },
  "hooks": {
    "post-edit": "~/.bmad/packages/doc-automation/hooks/post-edit-hook.sh"
  }
}
```

## 🔄 Automation Workflows

### Real-Time File Processing
When you edit a markdown file:
1. **Post-edit hook triggers** automatically
2. **File health assessed** (size, structure, links)
3. **Auto-fixes applied** if enabled (missing headings, formatting)
4. **Links validated** and broken links reported
5. **Health cache updated** for status line
6. **Results displayed** with colored output

### Background Health Monitoring
The daemon service continuously:
1. **Monitors file changes** using filesystem watchers
2. **Updates health scores** when files are modified
3. **Maintains link cache** for performance
4. **Triggers periodic scans** for comprehensive health checks
5. **Updates status line** in real-time

### Scheduled Maintenance
Set up automated maintenance with:
```bash
# Daily health checks
bmad schedule check-doc-health --daily

# Weekly link validation
bmad schedule validate-doc-links --weekly

# Monthly standards enforcement
bmad schedule enforce-doc-standards --monthly

# Weekly sitemap regeneration
bmad schedule generate-doc-sitemap --weekly
```

## 🛠️ Advanced Usage

### Custom Health Scoring
```bash
# Custom file size limits
bmad run check-doc-health custom_limits='{"warning": 8000, "critical": 12000}'

# Focus on specific directories
bmad run check-doc-health include_patterns='["docs/api/**", "docs/guides/**"]'

# Skip archived content
bmad run check-doc-health exclude_patterns='["docs/archive/**", "docs/legacy/**"]'
```

### Advanced Link Validation
```bash
# External links only with custom timeout
bmad run validate-doc-links external_only=true timeout_seconds=15

# Parallel processing optimization
bmad run validate-doc-links max_parallel=20

# Cache management
bmad run validate-doc-links clear_cache=true
```

### Professional Sitemap Generation
```bash
# Multiple output formats
bmad run generate-doc-sitemap output_formats='["markdown", "json", "html", "mermaid"]'

# Topic-based organization
bmad run generate-doc-sitemap generate_topics=true suggest_paths=true

# Custom sorting and grouping
bmad run generate-doc-sitemap sort_by=health group_by=topic
```

## 📈 Performance Optimization

### File System Watchers
- **macOS**: Uses `fswatch` for efficient file change detection
- **Linux**: Uses `inotify-tools` for low-overhead monitoring
- **Fallback**: Periodic scanning if watchers unavailable

### Link Validation Caching
- **Cache Duration**: 1 hour for successful external link checks
- **Cache Storage**: `/tmp/bmad-link-cache`
- **Performance**: 10x faster validation for repeated checks

### Parallel Processing
- **Link Validation**: Up to 10 concurrent external link checks
- **Health Analysis**: Parallel file processing for large document sets
- **Cross-Reference Generation**: Optimized metadata extraction

## 🔧 Troubleshooting

### Common Issues

#### Status Line Not Updating
```bash
# Check hook installation
bmad-doc-automation hooks status

# Reinstall hooks
bmad-doc-automation hooks install

# Test status line manually
~/.bmad/packages/doc-automation/hooks/doc-health-status-line.sh --test
```

#### File Watcher Not Working
```bash
# Install file system watcher
# macOS:
brew install fswatch

# Ubuntu/Debian:
sudo apt-get install inotify-tools

# CentOS/RHEL:
sudo yum install inotify-tools

# Restart daemon
bmad-doc-automation hooks install
```

#### Broken Links False Positives
```bash
# Update link validation settings
export LINK_TIMEOUT_SECONDS=30
export VALIDATE_EXTERNAL=false  # Disable external validation

# Clear link cache
rm -f /tmp/bmad-link-cache

# Re-run validation
bmad run validate-doc-links
```

### Debug Mode
```bash
# Enable debug output
export BMAD_DEBUG=true

# View detailed logs
tail -f /tmp/bmad-*.log

# Test individual components
~/.bmad/packages/doc-automation/hooks/post-edit-hook.sh test-file.md
```

## 🚀 Migration from Manual Processes

### From Manual Documentation Reviews
**Before**: Manual checking of documentation files, inconsistent standards  
**After**: Automated health monitoring, real-time status updates, consistent standards

### From Basic Link Checkers
**Before**: Simple link validation tools, no context or repair capabilities  
**After**: Intelligent link validation with caching, auto-repair, and detailed reporting

### From Template Copy-Paste
**Before**: Copying documentation templates manually, inconsistent formatting  
**After**: Professional templates with BMAD integration, automatic application

### Migration Steps
1. **Install Package**: Run one-click installer
2. **Initial Health Check**: `bmad run check-doc-health`
3. **Fix Critical Issues**: Use auto-fix or agent guidance
4. **Enable Automation**: Install hooks for real-time monitoring
5. **Apply Templates**: Standardize existing documentation
6. **Schedule Maintenance**: Set up automated maintenance tasks

## 🔮 Future Enhancements

### Planned Features
- **Multi-Language Support**: Templates and validation for multiple languages
- **Visual Dashboard**: Web-based health monitoring dashboard
- **Git Integration**: Pre-commit hooks and PR validation
- **Team Collaboration**: Shared health metrics and team dashboards
- **AI Enhancement**: LLM-powered content analysis and suggestions

### Extension Points
- **Custom Templates**: Create project-specific documentation templates
- **Custom Validators**: Add domain-specific validation rules
- **Integration APIs**: Connect with external documentation tools
- **Custom Agents**: Develop specialized BMAD agents for specific needs

## 📞 Support & Community

### Getting Help
- **Documentation**: Complete package documentation in README.md
- **Examples**: Working examples in each template and task
- **Troubleshooting**: Comprehensive troubleshooting guide included
- **Debug Tools**: Built-in debug mode and logging

### Contributing
- **Bug Reports**: Submit issues with detailed reproduction steps
- **Feature Requests**: Propose enhancements with use cases
- **Template Contributions**: Share professional templates
- **Script Improvements**: Contribute automation script enhancements

### Community
- **Discord**: Join the BMAD community for support and discussions
- **GitHub**: Contribute to the open-source BMAD framework
- **Blog Posts**: Share your documentation automation success stories

---

## 📄 License

This package is distributed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **BMAD Framework**: Built on the powerful BMAD automation framework
- **Claude Code**: Seamless integration with Claude Code editor
- **Community Contributors**: Thanks to all community members who provided feedback and testing

---

**Generated by**: BMAD Documentation Automation Package v1.0.0  
**Last Updated**: $(date)  
**Package Homepage**: https://bmad.dev/packages/doc-automation
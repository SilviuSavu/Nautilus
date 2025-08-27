# BMAD Documentation Automation Examples

**Package**: doc-automation v1.0.0  
**Purpose**: Real-world usage examples and integration patterns

## 📚 Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [BMAD Framework Integration](#bmad-framework-integration)
3. [Claude Code Integration](#claude-code-integration)
4. [Automation Workflows](#automation-workflows)
5. [Template Application](#template-application)
6. [Advanced Scenarios](#advanced-scenarios)
7. [Team Workflows](#team-workflows)
8. [CI/CD Integration](#cicd-integration)

---

## 🚀 Basic Usage Examples

### Example 1: First-Time Health Check
```bash
# Scenario: You've just installed the package and want to assess your documentation health
$ bmad-doc-automation health

📊 BMAD Documentation Health Analysis
====================================
Overall Health Score: 87% ✅ GOOD
Total Files: 23
Issues Found: 3 minor

File Breakdown:
✅ Excellent (< 10KB): 18 files (78%)
⚠️  Large (10-15KB): 3 files (13%)
❌ Oversized (> 15KB): 2 files (9%)

Recommended Actions:
• Split 2 oversized files for Claude Code compatibility
• Fix 1 broken internal link in docs/api/README.md
• Add H1 heading to docs/guides/getting-started.md

💡 Next Steps:
  - Run: bmad run enforce-doc-standards auto_fix=true
  - Fix oversized files: docs/api/complete-reference.md (18.2KB)
  - Validate links: bmad-doc-automation validate-links
```

### Example 2: Real-Time Status Monitoring
```bash
# Scenario: You want to see live documentation health in your Claude Code status bar
$ bmad-doc-automation status --colored

📚 Docs: ✅ 87% (good)

# Install automatic updates
$ bmad-doc-automation hooks install

✅ Status Line Script: Working
✅ Post-Edit Hook: Installed 
✅ Daemon Service: LaunchAgent running
✅ Claude Code Integration: Post-edit hook installed
```

### Example 3: Link Validation
```bash
# Scenario: You want to validate all links in your documentation
$ bmad-doc-automation validate-links

🔗 BMAD Link Validation Results
==============================
Total Files Scanned: 23
Total Links Found: 157
  External Links: 43
  Internal Links: 89
  Anchor Links: 25
Broken Links: 3
Link Health Score: 98% 🎉 EXCELLENT

🚨 Broken Links Details:
- ❌ EXTERNAL: https://old-api.example.com/docs (HTTP 404)
- 🔗 INTERNAL: docs/api/deprecated.md (file not found)
- ⚠️  ANCHOR: docs/architecture/overview.md#database-section (anchor not found)

💡 BMAD Quick Actions:
  - bmad agent doc-health fix-broken-links    # Interactive repair
  - bmad run validate-doc-links auto_fix=true # Automated fixes
```

---

## 🤖 BMAD Framework Integration

### Example 4: Using BMAD Tasks
```bash
# Scenario: Using the full BMAD framework for enhanced functionality

# Check documentation health with detailed reporting
$ bmad run check-doc-health detailed=true

📋 Analyzing current documentation standards...
   📊 Scanning 23 files for compliance
   📋 Checking against enterprise standards profile
   
   Current Compliance:
   - File naming: 21/23 (91.3%) compliant
   - Size limits: 21/23 (91.3%) compliant  
   - Structure: 22/23 (95.7%) compliant
   - Formatting: 20/23 (87.0%) compliant
   
   Overall Standards Score: 91.1% ✅ EXCELLENT

# Enforce standards with automatic fixes
$ bmad run enforce-doc-standards auto_fix=true

🔧 Applying automatic corrections...
   
   ✅ Fixed file naming issues:
   - User Guide.md → user-guide.md
   - API_Reference.md → api-reference.md
   
   ✅ Corrected heading hierarchy:
   - Fixed 2 documents with multiple H1 headings
   - Added missing H1 heading to 1 document
   
   ✅ Enhanced code blocks:
   - Added language specs to 12 code blocks
   - Fixed indentation in 5 code examples

# Generate comprehensive sitemap
$ bmad run generate-doc-sitemap generate_topics=true

🗺️  Discovering documentation structure...
   📁 Scanning directories: 8 folders
   📄 Found files: 23 markdown files
   📊 Total size: 487 KB
   🏷️  Extracted topics: 12 unique topics
   
   ✅ Markdown sitemap: docs/automation/sitemap.md
   ✅ JSON structure: docs/automation/sitemap.json
   ✅ HTML sitemap: docs/automation/sitemap.html
   ✅ Statistics: 23 files categorized and mapped
```

### Example 5: Interactive Agent Usage
```bash
# Scenario: Using Dr. DocHealth agent for interactive problem solving

$ bmad agent doc-health analyze

🏥 Dr. DocHealth - Documentation Health Analysis
===============================================

Hello! I'm Dr. DocHealth, your documentation wellness specialist. 
I've analyzed your documentation and found some areas for improvement.

📊 Health Summary:
   - Overall Score: 87% (Good, but room for improvement)
   - Critical Issues: 2 files need immediate attention
   - Opportunities: 5 optimization suggestions

🚨 Critical Issues Requiring Attention:

1. **docs/api/complete-reference.md** (18.2 KB)
   - Issue: File too large for Claude Code compatibility
   - Impact: Cannot be read by Claude Code's Read tool
   - Recommendation: Split into focused sections
   
2. **docs/deployment/enterprise-setup.md** (17.1 KB)  
   - Issue: Approaching size limit
   - Impact: Difficult to navigate and edit
   - Recommendation: Modularize deployment steps

Would you like me to help you fix these critical issues? [Y/n]: Y

🔧 Let's start with the API reference document...

I can help you split this large file in several ways:
1. 📋 By API endpoints (Authentication, Users, Orders, etc.)
2. 🏗️  By functionality (Core API, Admin API, Webhooks)  
3. 📖 By audience (Developer Guide, Admin Guide, Reference)

Which approach would work best for your use case? [1-3]: 1

Excellent choice! I'll help you create focused API documentation files:
- docs/api/authentication.md
- docs/api/users.md  
- docs/api/orders.md
- docs/api/webhooks.md
- docs/api/README.md (overview and navigation)

Proceeding with restructuring... ✅ Done!

📋 Updated cross-references in 5 related files
🎯 All documents now meet enterprise standards
```

---

## 💻 Claude Code Integration

### Example 6: Status Line Integration
```bash
# Scenario: Automatic health monitoring in Claude Code status bar

# After installation, your Claude Code status bar shows:
📚 ✅ 87%

# When you edit a file, the status updates in real-time:
# Before edit: 📚 ✅ 87%
# After edit:  📚 ✅ 89%  (if improvements made)
# Or:          📚 ⚠️ 85%   (if issues introduced)
```

### Example 7: Post-Edit Hook in Action
```bash
# Scenario: You edit a markdown file in Claude Code

# You save docs/new-feature.md with content:
"""
This is about our new feature.

Here's a [broken link](docs/nonexistent.md).
"""

# Automatically triggered:
BMAD Doc Health: docs/new-feature.md (no_heading)
  ⚠️ Found 1 broken link

# With auto-fix enabled:
BMAD Doc Health: docs/new-feature.md (excellent)  
  ✅ Applied 1 auto-fix (added H1 heading)
  ⚠️ Found 1 broken link

# Final file content:
"""
# New Feature

This is about our new feature.

Here's a [broken link](docs/nonexistent.md).
"""
```

### Example 8: Claude Code Settings Configuration
```json
// ~/.claude/settings.json
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

---

## ⚙️ Automation Workflows

### Example 9: Scheduled Maintenance
```bash
# Scenario: Setting up automated documentation maintenance

# Daily health monitoring  
$ bmad schedule check-doc-health --daily
✅ Scheduled daily health checks at 9:00 AM

# Weekly link validation
$ bmad schedule validate-doc-links --weekly  
✅ Scheduled weekly link validation on Sundays at 10:00 AM

# Monthly standards enforcement
$ bmad schedule enforce-doc-standards --monthly
✅ Scheduled monthly standards check on the 1st at 11:00 AM

# View scheduled tasks
$ bmad schedule list

Scheduled BMAD Tasks:
┌─────────────────────┬─────────────┬──────────────────────┬─────────┐
│ Task                │ Frequency   │ Next Run             │ Status  │
├─────────────────────┼─────────────┼──────────────────────┼─────────┤
│ check-doc-health    │ Daily       │ Tomorrow 9:00 AM     │ Active  │
│ validate-doc-links  │ Weekly      │ Sunday 10:00 AM      │ Active  │
│ enforce-doc-standards│ Monthly    │ Dec 1st 11:00 AM     │ Active  │
└─────────────────────┴─────────────┴──────────────────────┴─────────┘
```

### Example 10: File Watcher Automation
```bash
# Scenario: Real-time processing with file system watchers

# Install file watcher daemon
$ bmad-doc-automation hooks install

Installing file watcher daemon...
✅ fswatch-based file watcher installed
✅ File watcher started (PID: 12345)

# Now when you:
# 1. Create docs/new-guide.md
# 2. Edit existing files  
# 3. Delete documentation files

# The system automatically:
# - Validates the changes
# - Updates health scores
# - Applies auto-fixes (if enabled)
# - Updates status line
# - Logs all activities

# View real-time logs:
$ tail -f /tmp/bmad-status-line.log

[2025-08-26 09:15:23] [doc-health] [INFO] File change detected: docs/new-guide.md (create)
[2025-08-26 09:15:23] [doc-health] [INFO] Applying auto-fixes to docs/new-guide.md
[2025-08-26 09:15:23] [doc-health] [INFO] Applied 1 auto-fix (added H1 heading)
[2025-08-26 09:15:24] [doc-health] [INFO] Health cache updated: 89% (good)
```

---

## 📋 Template Application

### Example 11: API Documentation Template
```bash
# Scenario: Creating API documentation for a new microservice

$ bmad apply template api-documentation target=docs/api/user-service.md

🔄 Applying template 'api-documentation' to docs/api/user-service.md...

✅ Template applied successfully!

📋 Generated sections:
- Overview with service description
- Quick Reference with base URL and key endpoints  
- Authentication setup with examples
- Core endpoint documentation with request/response examples
- Error handling and status codes
- SDK examples for multiple languages
- Testing and troubleshooting guides

📝 Next steps:
1. Update base URL: https://api.yourcompany.com/v1
2. Add your specific endpoints to replace examples
3. Update authentication method (API key, OAuth, etc.)
4. Add real request/response examples
5. Test all code examples

⚡ File ready for customization: docs/api/user-service.md
```

### Example 12: Architecture Documentation
```bash
# Scenario: Documenting a new microservices architecture

$ bmad apply template architecture-document target=docs/architecture/microservices-platform.md

🔄 Applying template 'architecture-document' to docs/architecture/microservices-platform.md...

✅ Template applied successfully!

📋 Generated comprehensive architecture documentation:
- High-level architecture with Mermaid diagrams
- System components with detailed descriptions
- Technology stack specifications
- Data flow and communication patterns
- Security architecture and considerations
- Scalability and performance requirements
- Deployment architecture
- Migration strategy and timeline
- Risk assessment and mitigation

📝 Customization checklist:
□ Replace placeholder technology stack with actual choices
□ Update system diagrams with real service names
□ Add specific performance targets and SLAs
□ Include actual security requirements and compliance needs
□ Update migration timeline with real dates
□ Add project-specific risks and mitigation strategies

🎯 Pro tip: Use 'bmad run generate-doc-sitemap' after customization to create cross-references
```

### Example 13: Multiple Template Application
```bash
# Scenario: Setting up documentation for a new project

# Apply multiple templates for comprehensive documentation
$ bmad apply template api-documentation target=docs/api/README.md
✅ API documentation template applied

$ bmad apply template architecture-document target=docs/architecture/system-design.md  
✅ Architecture template applied

$ bmad apply template deployment-guide target=docs/deployment/production-setup.md
✅ Deployment guide template applied

$ bmad apply template troubleshooting-guide target=docs/support/common-issues.md
✅ Troubleshooting template applied

# Generate sitemap to connect everything
$ bmad run generate-doc-sitemap generate_topics=true

🗺️ Generated comprehensive documentation structure:
- Cross-references between all documents
- Topic-based navigation (API, Architecture, Deployment, Support)
- Suggested reading paths for different user types
- Quick navigation links

📈 Documentation health improved from 45% to 94%!
```

---

## 🔬 Advanced Scenarios

### Example 14: Custom Health Scoring
```bash
# Scenario: Customizing health criteria for your team's needs

# Custom file size limits for your specific use case
$ bmad run check-doc-health custom_limits='{
  "warning": 8000,
  "critical": 12000,
  "excellent": 6000
}'

📊 Custom Health Analysis
========================
Using custom file size limits:
- Excellent: < 6KB
- Good: 6-8KB  
- Warning: 8-12KB
- Critical: > 12KB

Results with custom criteria:
Overall Health Score: 91% ✅ EXCELLENT
File Distribution:
✅ Excellent: 15 files (65%)
✅ Good: 6 files (26%)
⚠️  Warning: 2 files (9%)
❌ Critical: 0 files (0%)

🎯 Your team's shorter file preference shows excellent compliance!
```

### Example 15: Multi-Language Documentation
```bash
# Scenario: Managing documentation in multiple languages

# English documentation health
$ bmad run check-doc-health include_patterns='["docs/en/**"]'
Overall Health: 94% ✅ EXCELLENT (English docs)

# Spanish documentation health  
$ bmad run check-doc-health include_patterns='["docs/es/**"]'
Overall Health: 78% ⚠️ ATTENTION NEEDED (Spanish docs)

# Generate language-specific sitemaps
$ bmad run generate-doc-sitemap include_patterns='["docs/en/**"]' output_file="sitemap-en.md"
$ bmad run generate-doc-sitemap include_patterns='["docs/es/**"]' output_file="sitemap-es.md"

# Cross-language link validation
$ bmad run validate-doc-links include_patterns='["docs/**"]' validate_cross_language=true

🌐 Multi-language validation results:
- English-Spanish cross-references: 12 links validated
- Missing translations detected: 3 English docs without Spanish versions
- Broken cross-language links: 1 (docs/es/api/usuarios.md → docs/en/api/users.md)
```

### Example 16: Large-Scale Documentation Management
```bash
# Scenario: Managing documentation for a large enterprise project (500+ files)

# Parallel processing for large doc sets
$ bmad run check-doc-health max_parallel=20 batch_size=50

📊 Large-Scale Documentation Analysis
=====================================
Processing 547 files in batches...

Batch 1/11: Processing files 1-50... ✅ Complete (2.3s)
Batch 2/11: Processing files 51-100... ✅ Complete (2.1s)
Batch 3/11: Processing files 101-150... ✅ Complete (2.4s)
...
Batch 11/11: Processing files 501-547... ✅ Complete (1.8s)

Total Processing Time: 24.6 seconds
Overall Health Score: 89% ✅ GOOD

Performance Optimizations Applied:
✅ Parallel file processing (20 concurrent)
✅ Batch processing (50 files per batch)
✅ Efficient memory usage
✅ Progress reporting

Large File Issues (> 15KB):
- 23 files require splitting for Claude Code compatibility
- Affected areas: API docs (12), Architecture (8), Deployment (3)

💡 Recommendation: Use 'bmad run enforce-doc-standards restructure_content=true' for automatic splitting
```

---

## 👥 Team Workflows

### Example 17: Onboarding New Team Members
```bash
# Scenario: Setting up BMAD for a new developer

# Quick team setup script
#!/bin/bash
# File: setup-bmad-for-team-member.sh

echo "🚀 Setting up BMAD Documentation Automation for $(whoami)"

# Install package
curl -fsSL https://bmad.dev/install/doc-automation | bash

# Configure team-specific settings
export BMAD_AUTO_FIX=true
export BMAD_VALIDATE_ON_EDIT=true
export DOC_HEALTH_WARNING_SIZE=8000  # Team prefers smaller files

# Install hooks with team configuration
bmad-doc-automation hooks install

# Initial health check
echo "📊 Initial documentation health check..."
bmad-doc-automation health

# Show team dashboard URL
echo "📈 Team Documentation Dashboard: https://docs.yourcompany.com/health"
echo "✅ Setup complete! Documentation will now be automatically monitored."
```

### Example 18: Team Health Dashboard
```bash
# Scenario: Monitoring team-wide documentation health

# Generate team health report
$ bmad run check-doc-health generate_team_report=true

📊 Team Documentation Health Dashboard
======================================
Report Period: Last 30 days
Team: Platform Engineering (12 members)

Overall Team Metrics:
📈 Health Trend: +7% improvement (82% → 89%)
📝 Files Added: 23 new documentation files
🔧 Auto-fixes Applied: 156 automatic improvements
🔗 Links Validated: 1,247 links checked

Individual Contributions:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│ Team Member     │ Files Added │ Health Δ    │ Auto-fixes  │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Alice (Tech Lead│ 8           │ +12%        │ 45          │
│ Bob (Backend)   │ 5           │ +8%         │ 23          │
│ Charlie (DevOps)│ 6           │ +15%        │ 34          │
│ Diana (Frontend)│ 4           │ +5%         │ 19          │
└─────────────────┴─────────────┴─────────────┴─────────────┘

🎯 Team Goals Progress:
✅ Maintain >85% health score: 89% (Goal exceeded!)
✅ <5% broken links: 2.1% (Goal achieved!)
⚠️  All files <15KB: 91% (9% still oversized)

📋 Action Items:
- Split 3 remaining oversized API documentation files
- Add deployment guides for new microservices  
- Update architecture docs for recent system changes
```

### Example 19: Code Review Integration
```bash
# Scenario: Documentation review process for pull requests

# Pre-commit hook for documentation validation
# File: .git/hooks/pre-commit
#!/bin/bash

echo "🔍 BMAD Documentation Validation"
echo "================================="

# Check if any markdown files are being committed
if git diff --cached --name-only | grep -q '\.md$'; then
    echo "📝 Markdown files detected, running documentation validation..."
    
    # Validate changed documentation files
    for file in $(git diff --cached --name-only | grep '\.md$'); do
        if [ -f "$file" ]; then
            echo "  Checking: $file"
            
            # Run post-edit hook for validation
            ~/.bmad/packages/doc-automation/hooks/post-edit-hook.sh "$file" "commit-check"
            
            if [ $? -ne 0 ]; then
                echo "❌ Documentation validation failed for: $file"
                echo "💡 Fix issues and commit again, or skip with --no-verify"
                exit 1
            fi
        fi
    done
    
    # Overall health check  
    health_score=$(bmad-doc-automation status | grep -o '[0-9]\+%' | sed 's/%//')
    if [ "$health_score" -lt 80 ]; then
        echo "❌ Overall documentation health too low: ${health_score}%"
        echo "💡 Improve documentation health before committing"
        exit 1
    fi
    
    echo "✅ All documentation validation checks passed!"
else
    echo "ℹ️  No markdown files changed, skipping documentation validation"
fi

# Continue with normal commit process
exit 0
```

---

## 🔄 CI/CD Integration

### Example 20: GitHub Actions Integration
```yaml
# File: .github/workflows/documentation-health.yml

name: Documentation Health Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
    paths: [ '**/*.md' ]

jobs:
  doc-health:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Install BMAD Doc Automation
      run: |
        curl -fsSL https://bmad.dev/install/doc-automation | bash
        echo "$HOME/.local/bin" >> $GITHUB_PATH
        
    - name: Check Documentation Health  
      run: |
        bmad-doc-automation health
        health_score=$(bmad-doc-automation status | grep -o '[0-9]\+' | head -1)
        echo "HEALTH_SCORE=$health_score" >> $GITHUB_ENV
        
        if [ "$health_score" -lt 85 ]; then
          echo "❌ Documentation health score too low: ${health_score}%"
          exit 1
        fi
        
    - name: Validate Links
      run: |
        bmad-doc-automation validate-links
        if [ $? -eq 2 ]; then
          echo "❌ Critical link validation issues found"
          exit 1
        fi
        
    - name: Generate Health Report
      if: always()
      run: |
        bmad run check-doc-health detailed=true > doc-health-report.md
        
    - name: Upload Health Report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: documentation-health-report
        path: doc-health-report.md
        
    - name: Comment PR with Health Status
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const healthScore = process.env.HEALTH_SCORE;
          const icon = healthScore >= 95 ? '🎉' : healthScore >= 85 ? '✅' : healthScore >= 70 ? '⚠️' : '❌';
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## 📚 Documentation Health Check\n\n${icon} **Health Score**: ${healthScore}%\n\nFull report available in workflow artifacts.`
          });
```

### Example 21: Jenkins Pipeline Integration
```groovy
// File: Jenkinsfile
pipeline {
    agent any
    
    environment {
        BMAD_ENABLED = 'true'
        BMAD_AUTO_FIX = 'false'  // Disable auto-fixes in CI
    }
    
    stages {
        stage('Install BMAD Doc Automation') {
            steps {
                script {
                    sh '''
                        curl -fsSL https://bmad.dev/install/doc-automation | bash
                        export PATH="$HOME/.local/bin:$PATH"
                    '''
                }
            }
        }
        
        stage('Documentation Health Check') {
            steps {
                script {
                    def healthResult = sh(
                        script: 'bmad-doc-automation health',
                        returnStatus: true
                    )
                    
                    def healthScore = sh(
                        script: 'bmad-doc-automation status | grep -o "[0-9]\\+" | head -1',
                        returnStdout: true
                    ).trim()
                    
                    echo "Documentation Health Score: ${healthScore}%"
                    
                    if (healthScore.toInteger() < 85) {
                        currentBuild.result = 'UNSTABLE'
                        echo "⚠️ Documentation health below threshold"
                    }
                }
            }
        }
        
        stage('Link Validation') {
            steps {
                script {
                    def linkResult = sh(
                        script: 'bmad-doc-automation validate-links',
                        returnStatus: true
                    )
                    
                    if (linkResult == 2) {
                        error "❌ Critical link validation issues found"
                    } else if (linkResult == 1) {
                        currentBuild.result = 'UNSTABLE' 
                        echo "⚠️ Some link validation issues found"
                    }
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    bmad run check-doc-health detailed=true > doc-health-report.md
                    bmad run generate-doc-sitemap > doc-sitemap-report.md
                '''
                
                archiveArtifacts artifacts: '*-report.md', allowEmptyArchive: true
            }
        }
    }
    
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'doc-health-report.md',
                reportName: 'Documentation Health Report'
            ])
        }
        
        failure {
            emailext (
                subject: "Documentation Health Check Failed - ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: """
                    The documentation health check has failed.
                    
                    Build: ${env.BUILD_URL}
                    Health Reports: ${env.BUILD_URL}artifact/
                    
                    Please review and fix documentation issues before merging.
                """,
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Example 22: Docker Integration for CI/CD
```dockerfile
# File: Dockerfile.doc-automation
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    git \
    inotify-tools \
    python3 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install BMAD Documentation Automation
RUN curl -fsSL https://bmad.dev/install/doc-automation | bash

# Create workspace
WORKDIR /workspace

# Set environment variables
ENV BMAD_ENABLED=true
ENV BMAD_AUTO_FIX=false
ENV BMAD_VALIDATE_ON_EDIT=false

# Entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["health"]
```

```bash
# File: docker-entrypoint.sh
#!/bin/bash
set -e

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

case "$1" in
    "health")
        echo "🔍 Running documentation health check..."
        bmad-doc-automation health
        ;;
    "validate-links")
        echo "🔗 Validating documentation links..."
        bmad-doc-automation validate-links
        ;;
    "generate-sitemap") 
        echo "🗺️ Generating documentation sitemap..."
        bmad-doc-automation generate-sitemap
        ;;
    "full-check")
        echo "🚀 Running comprehensive documentation check..."
        bmad-doc-automation health
        bmad-doc-automation validate-links
        bmad-doc-automation generate-sitemap
        ;;
    *)
        echo "Usage: docker run doc-automation [health|validate-links|generate-sitemap|full-check]"
        exit 1
        ;;
esac
```

```bash
# Usage in CI/CD pipeline
docker build -t doc-automation -f Dockerfile.doc-automation .
docker run --rm -v $(pwd):/workspace doc-automation full-check
```

---

## 🎯 Success Metrics

### Example 23: Measuring Documentation Improvement
```bash
# Before BMAD implementation:
Documentation Health: 34% ❌ CRITICAL
- 67% of files oversized (>15KB) 
- 23% broken links
- Inconsistent formatting across teams
- No automated maintenance
- Manual health checks taking 2-3 hours weekly

# After 1 month with BMAD:
$ bmad-doc-automation health

Documentation Health: 91% ✅ EXCELLENT  
- 91% of files optimal size (<10KB)
- <2% broken links  
- Consistent formatting automatically enforced
- Zero manual maintenance time
- Real-time health monitoring

📈 Improvement Metrics:
• Health Score: +167% improvement (34% → 91%)
• Time Saved: 2-3 hours/week → 0 hours/week (automated)
• Developer Satisfaction: +40% (internal survey)
• Documentation Usage: +156% increase in internal docs usage
• Onboarding Time: -50% for new team members
```

This comprehensive examples guide demonstrates the power and flexibility of the BMAD Documentation Automation package across various real-world scenarios. From basic usage to advanced enterprise deployments, these examples show how to transform documentation management from a manual burden into an automated, intelligent system that improves team productivity and documentation quality.

---

**Last Updated**: August 26, 2025  
**Examples Version**: 1.0.0  
**Package**: BMAD Documentation Automation v1.0.0
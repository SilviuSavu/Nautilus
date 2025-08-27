#!/bin/bash
# BMAD Documentation Automation Package Installer
# One-click installation of enterprise documentation automation system

set -e

# Version and Package Info
PACKAGE_NAME="doc-automation"
PACKAGE_VERSION="1.0.0"
BMAD_REPO="https://github.com/bmad-framework/bmad-packages"  # Replace with actual repo
INSTALL_DIR="$HOME/.bmad/packages/$PACKAGE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Installation log
INSTALL_LOG="/tmp/bmad-doc-automation-install.log"

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [BMAD-INSTALLER] [$level] $message" >> "$INSTALL_LOG"
    
    case "$level" in
        "ERROR") echo -e "${RED}❌ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️ $message${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️ $message${NC}" ;;
        "STEP") echo -e "${CYAN}🔄 $message${NC}" ;;
        *) echo "$message" ;;
    esac
}

# Print banner
print_banner() {
    echo -e "${BOLD}${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    🚀 BMAD Documentation Automation Package Installer 🚀         ║
║                                                                  ║
║    Enterprise-grade automatic documentation system              ║
║    with real-time health monitoring and intelligent fixes       ║
║                                                                  ║
║    Version: 1.0.0                                               ║
║    Package: doc-automation                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log "STEP" "Checking system requirements"
    
    local missing_deps=()
    local optional_deps=()
    
    # Required dependencies
    if ! command -v bash >/dev/null 2>&1; then
        missing_deps+=("bash")
    fi
    
    if ! command -v git >/dev/null 2>&1; then
        missing_deps+=("git")
    fi
    
    if ! command -v curl >/dev/null 2>&1; then
        missing_deps+=("curl")
    fi
    
    # Optional but recommended dependencies
    if ! command -v fswatch >/dev/null 2>&1 && ! command -v inotifywait >/dev/null 2>&1; then
        optional_deps+=("fswatch or inotify-tools")
    fi
    
    if ! command -v python3 >/dev/null 2>&1; then
        optional_deps+=("python3")
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        optional_deps+=("jq")
    fi
    
    # Report results
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "ERROR" "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        echo -e "${YELLOW}Install required dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            case "$dep" in
                "git")
                    echo "  • macOS: brew install git"
                    echo "  • Ubuntu/Debian: sudo apt-get install git"
                    echo "  • CentOS/RHEL: sudo yum install git"
                    ;;
                "curl")
                    echo "  • macOS: brew install curl (usually pre-installed)"
                    echo "  • Ubuntu/Debian: sudo apt-get install curl"
                    echo "  • CentOS/RHEL: sudo yum install curl"
                    ;;
            esac
        done
        return 1
    fi
    
    if [ ${#optional_deps[@]} -gt 0 ]; then
        log "WARN" "Optional dependencies missing: ${optional_deps[*]}"
        log "INFO" "Some features will be limited without these dependencies"
        echo ""
        echo -e "${YELLOW}Recommended installations:${NC}"
        for dep in "${optional_deps[@]}"; do
            case "$dep" in
                "fswatch or inotify-tools")
                    echo "  • macOS: brew install fswatch"
                    echo "  • Ubuntu/Debian: sudo apt-get install inotify-tools"
                    echo "  • CentOS/RHEL: sudo yum install inotify-tools"
                    ;;
                "python3")
                    echo "  • macOS: brew install python3"
                    echo "  • Ubuntu/Debian: sudo apt-get install python3"
                    echo "  • CentOS/RHEL: sudo yum install python3"
                    ;;
                "jq")
                    echo "  • macOS: brew install jq"
                    echo "  • Ubuntu/Debian: sudo apt-get install jq"
                    echo "  • CentOS/RHEL: sudo yum install jq"
                    ;;
            esac
        done
        echo ""
    fi
    
    # Check BMAD framework
    if [ ! -d "$HOME/.bmad" ]; then
        log "WARN" "BMAD framework not found at $HOME/.bmad"
        log "INFO" "This package will work standalone but some features may be limited"
    else
        log "SUCCESS" "BMAD framework detected"
    fi
    
    log "SUCCESS" "System requirements check completed"
    return 0
}

# Detect installation method
detect_installation_method() {
    log "STEP" "Detecting installation method"
    
    if [ -f "$(dirname "$0")/manifest.yaml" ]; then
        # Local installation from package directory
        INSTALL_METHOD="local"
        PACKAGE_SOURCE="$(dirname "$0")"
        log "INFO" "Local installation detected: $PACKAGE_SOURCE"
    elif [ -n "${BMAD_PACKAGE_SOURCE:-}" ]; then
        # Installation via BMAD package manager
        INSTALL_METHOD="bmad"
        PACKAGE_SOURCE="$BMAD_PACKAGE_SOURCE"
        log "INFO" "BMAD package manager installation"
    else
        # Remote installation - download package
        INSTALL_METHOD="remote"
        PACKAGE_SOURCE="$BMAD_REPO"
        log "INFO" "Remote installation from: $PACKAGE_SOURCE"
    fi
}

# Download package (for remote installation)
download_package() {
    if [ "$INSTALL_METHOD" != "remote" ]; then
        return 0
    fi
    
    log "STEP" "Downloading BMAD doc-automation package"
    
    local temp_dir="/tmp/bmad-doc-automation-$$"
    mkdir -p "$temp_dir"
    
    # Download package
    if git clone "$PACKAGE_SOURCE" "$temp_dir" 2>/dev/null; then
        PACKAGE_SOURCE="$temp_dir/packages/doc-automation"
        log "SUCCESS" "Package downloaded successfully"
    else
        log "ERROR" "Failed to download package from $PACKAGE_SOURCE"
        log "INFO" "Falling back to curl download"
        
        # Fallback to direct file download
        local download_url="$PACKAGE_SOURCE/archive/main.zip"
        if curl -L "$download_url" -o "$temp_dir/package.zip" && \
           unzip -q "$temp_dir/package.zip" -d "$temp_dir"; then
            PACKAGE_SOURCE="$temp_dir/*/packages/doc-automation"
            log "SUCCESS" "Package downloaded via fallback method"
        else
            log "ERROR" "Failed to download package"
            rm -rf "$temp_dir"
            return 1
        fi
    fi
    
    if [ ! -f "$PACKAGE_SOURCE/manifest.yaml" ]; then
        log "ERROR" "Invalid package structure - manifest.yaml not found"
        rm -rf "$temp_dir"
        return 1
    fi
}

# Create installation directory
create_install_directory() {
    log "STEP" "Creating installation directory"
    
    if [ -d "$INSTALL_DIR" ]; then
        log "WARN" "Installation directory already exists: $INSTALL_DIR"
        log "INFO" "Creating backup of existing installation"
        
        local backup_dir="${INSTALL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
        mv "$INSTALL_DIR" "$backup_dir"
        log "SUCCESS" "Existing installation backed up to: $backup_dir"
    fi
    
    mkdir -p "$INSTALL_DIR"
    log "SUCCESS" "Installation directory created: $INSTALL_DIR"
}

# Install package files
install_package_files() {
    log "STEP" "Installing package files"
    
    if [ ! -d "$PACKAGE_SOURCE" ]; then
        log "ERROR" "Package source directory not found: $PACKAGE_SOURCE"
        return 1
    fi
    
    # Copy all package files
    cp -r "$PACKAGE_SOURCE"/* "$INSTALL_DIR/"
    
    # Make scripts executable
    find "$INSTALL_DIR" -name "*.sh" -type f -exec chmod +x {} \;
    
    # Set proper permissions
    chmod 755 "$INSTALL_DIR"
    
    log "SUCCESS" "Package files installed to: $INSTALL_DIR"
}

# Install BMAD integration
install_bmad_integration() {
    log "STEP" "Installing BMAD integration"
    
    local bmad_dir="$HOME/.bmad"
    
    if [ ! -d "$bmad_dir" ]; then
        log "INFO" "Creating BMAD directory structure"
        mkdir -p "$bmad_dir/packages"
        mkdir -p "$bmad_dir/agents"
        mkdir -p "$bmad_dir/tasks"
        mkdir -p "$bmad_dir/templates"
    fi
    
    # Link agent
    if [ -f "$INSTALL_DIR/agents/doc-health.md" ]; then
        ln -sf "$INSTALL_DIR/agents/doc-health.md" "$bmad_dir/agents/"
        log "SUCCESS" "Agent 'doc-health' registered"
    fi
    
    # Link tasks
    for task_file in "$INSTALL_DIR/tasks"/*.md; do
        if [ -f "$task_file" ]; then
            ln -sf "$task_file" "$bmad_dir/tasks/"
            local task_name=$(basename "$task_file" .md)
            log "SUCCESS" "Task '$task_name' registered"
        fi
    done
    
    # Link templates
    for template_file in "$INSTALL_DIR/templates"/*.md; do
        if [ -f "$template_file" ]; then
            ln -sf "$template_file" "$bmad_dir/templates/"
            local template_name=$(basename "$template_file" .md)
            log "SUCCESS" "Template '$template_name' registered"
        fi
    done
    
    log "SUCCESS" "BMAD integration installed"
}

# Install hooks and automation
install_hooks() {
    log "STEP" "Installing hooks and automation"
    
    local hook_installer="$INSTALL_DIR/hooks/install-hooks.sh"
    
    if [ -f "$hook_installer" ]; then
        if "$hook_installer" install; then
            log "SUCCESS" "Hooks and automation installed"
        else
            log "WARN" "Hook installation completed with warnings"
            log "INFO" "Some features may not be available"
        fi
    else
        log "WARN" "Hook installer not found, skipping automatic hook installation"
        log "INFO" "You can manually install hooks later with: $INSTALL_DIR/hooks/install-hooks.sh"
    fi
}

# Create command-line interface
create_cli_interface() {
    log "STEP" "Creating command-line interface"
    
    local cli_script="/usr/local/bin/bmad-doc-automation"
    local user_cli_script="$HOME/.local/bin/bmad-doc-automation"
    
    # Try system-wide installation first
    if [ -w "/usr/local/bin" ] 2>/dev/null; then
        create_cli_script "$cli_script"
        log "SUCCESS" "CLI installed system-wide: $cli_script"
    else
        # Fall back to user installation
        mkdir -p "$HOME/.local/bin"
        create_cli_script "$user_cli_script"
        log "SUCCESS" "CLI installed for user: $user_cli_script"
        
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            log "INFO" "Add $HOME/.local/bin to your PATH for global access"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.bashrc"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.zshrc" 2>/dev/null || true
        fi
    fi
}

# Create CLI script
create_cli_script() {
    local cli_path="$1"
    
    cat > "$cli_path" << EOF
#!/bin/bash
# BMAD Documentation Automation CLI
# Generated by installer on $(date)

INSTALL_DIR="$INSTALL_DIR"

# Help function
show_help() {
    echo "BMAD Documentation Automation CLI"
    echo "================================="
    echo ""
    echo "Usage: bmad-doc-automation [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  health                    Check documentation health"
    echo "  validate-links           Validate all links"
    echo "  generate-sitemap         Generate documentation sitemap"
    echo "  enforce-standards        Enforce documentation standards"
    echo "  status                   Show real-time status"
    echo "  agent [COMMAND]          Interact with doc-health agent"
    echo "  hooks [install|status]   Manage hooks and automation"
    echo "  templates [list|apply]   Manage documentation templates"
    echo "  uninstall                Uninstall the package"
    echo "  help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  bmad-doc-automation health                    # Check health"
    echo "  bmad-doc-automation validate-links           # Validate links"
    echo "  bmad-doc-automation generate-sitemap         # Generate sitemap"
    echo "  bmad-doc-automation status --colored          # Colored status"
    echo "  bmad-doc-automation hooks install            # Install automation"
    echo ""
    echo "Package Location: \$INSTALL_DIR"
    echo "Version: $PACKAGE_VERSION"
}

# Main execution
case "\${1:-help}" in
    "health"|"check"|"check-health")
        "\$INSTALL_DIR/scripts/bmad-doc-maintenance.sh"
        ;;
    "validate-links"|"links")
        "\$INSTALL_DIR/scripts/bmad-link-validator.sh" "\${@:2}"
        ;;
    "generate-sitemap"|"sitemap")
        "\$INSTALL_DIR/scripts/bmad-cross-references.sh" "\${@:2}"
        ;;
    "enforce-standards"|"standards")
        echo "Standards enforcement task available via BMAD framework"
        echo "Run: bmad run enforce-doc-standards"
        ;;
    "status")
        if [ -f "\$INSTALL_DIR/hooks/doc-health-status-line.sh" ]; then
            "\$INSTALL_DIR/hooks/doc-health-status-line.sh" --status "\${@:2}"
        else
            echo "Status line not available - install hooks first"
        fi
        ;;
    "agent")
        echo "Agent interaction available via BMAD framework"
        echo "Run: bmad agent doc-health \${@:2}"
        ;;
    "hooks")
        if [ -f "\$INSTALL_DIR/hooks/install-hooks.sh" ]; then
            "\$INSTALL_DIR/hooks/install-hooks.sh" "\${@:2}"
        else
            echo "Hooks installer not found"
        fi
        ;;
    "templates")
        case "\${2:-list}" in
            "list")
                echo "Available templates:"
                ls "\$INSTALL_DIR/templates/"*.md 2>/dev/null | sed 's/.*\//  - /' | sed 's/\.md$//'
                ;;
            "apply")
                echo "Template application available via BMAD framework"
                echo "Run: bmad apply template \${3} target=\${4}"
                ;;
            *)
                echo "Usage: bmad-doc-automation templates [list|apply]"
                ;;
        esac
        ;;
    "uninstall")
        echo "Uninstalling BMAD Documentation Automation..."
        if [ -f "\$INSTALL_DIR/hooks/install-hooks.sh" ]; then
            "\$INSTALL_DIR/hooks/install-hooks.sh" uninstall
        fi
        rm -rf "\$INSTALL_DIR"
        rm -f "$cli_path"
        echo "✅ Package uninstalled successfully"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "Unknown command: \$1"
        echo "Run 'bmad-doc-automation help' for usage information"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$cli_path"
}

# Test installation
test_installation() {
    log "STEP" "Testing installation"
    
    local test_results=0
    
    # Test 1: Check package files
    if [ -f "$INSTALL_DIR/manifest.yaml" ]; then
        log "SUCCESS" "Package manifest found"
    else
        log "ERROR" "Package manifest missing"
        ((test_results++))
    fi
    
    # Test 2: Check scripts
    local scripts=("bmad-doc-maintenance.sh" "bmad-link-validator.sh" "bmad-cross-references.sh")
    for script in "${scripts[@]}"; do
        if [ -x "$INSTALL_DIR/scripts/$script" ]; then
            log "SUCCESS" "Script $script is executable"
        else
            log "ERROR" "Script $script is missing or not executable"
            ((test_results++))
        fi
    done
    
    # Test 3: Check agent
    if [ -f "$INSTALL_DIR/agents/doc-health.md" ]; then
        log "SUCCESS" "Doc-health agent found"
    else
        log "ERROR" "Doc-health agent missing"
        ((test_results++))
    fi
    
    # Test 4: Test status line
    if [ -x "$INSTALL_DIR/hooks/doc-health-status-line.sh" ]; then
        if "$INSTALL_DIR/hooks/doc-health-status-line.sh" --test > /dev/null 2>&1; then
            log "SUCCESS" "Status line integration working"
        else
            log "WARN" "Status line integration has issues"
        fi
    fi
    
    # Test 5: Test CLI
    if command -v bmad-doc-automation >/dev/null 2>&1; then
        log "SUCCESS" "CLI command available globally"
    else
        log "WARN" "CLI command not in PATH"
    fi
    
    if [ $test_results -eq 0 ]; then
        log "SUCCESS" "All installation tests passed"
        return 0
    else
        log "WARN" "Installation completed with $test_results issues"
        return 1
    fi
}

# Show usage instructions
show_usage_instructions() {
    echo ""
    echo -e "${BOLD}${GREEN}🎉 Installation Complete! 🎉${NC}"
    echo ""
    echo -e "${BOLD}Quick Start Guide:${NC}"
    echo -e "${CYAN}1. Check documentation health:${NC}"
    echo "   bmad-doc-automation health"
    echo ""
    echo -e "${CYAN}2. Show real-time status:${NC}"
    echo "   bmad-doc-automation status --colored"
    echo ""
    echo -e "${CYAN}3. Validate all links:${NC}"
    echo "   bmad-doc-automation validate-links"
    echo ""
    echo -e "${CYAN}4. Generate cross-references:${NC}"
    echo "   bmad-doc-automation generate-sitemap"
    echo ""
    echo -e "${BOLD}BMAD Framework Integration:${NC}"
    if [ -d "$HOME/.bmad" ]; then
        echo -e "${CYAN}• Run BMAD tasks:${NC}"
        echo "  bmad run check-doc-health"
        echo "  bmad run validate-doc-links"
        echo "  bmad run generate-doc-sitemap"
        echo "  bmad run enforce-doc-standards"
        echo ""
        echo -e "${CYAN}• Use interactive agent:${NC}"
        echo "  bmad agent doc-health analyze"
        echo "  bmad agent doc-health fix-critical"
        echo ""
        echo -e "${CYAN}• Apply templates:${NC}"
        echo "  bmad apply template api-documentation target=docs/api/new-service.md"
        echo "  bmad apply template architecture-document target=docs/architecture/system.md"
    else
        echo -e "${YELLOW}• Install BMAD framework for full integration${NC}"
    fi
    echo ""
    echo -e "${BOLD}Automation Features:${NC}"
    echo -e "${GREEN}✅ Real-time status line updates${NC}"
    echo -e "${GREEN}✅ Automatic file processing on edits${NC}"
    echo -e "${GREEN}✅ Background health monitoring${NC}"
    echo -e "${GREEN}✅ Link validation and auto-fixes${NC}"
    echo ""
    echo -e "${BOLD}Documentation:${NC}"
    echo "• Package location: $INSTALL_DIR"
    echo "• Installation log: $INSTALL_LOG"
    echo "• Help command: bmad-doc-automation help"
    echo ""
    echo -e "${CYAN}For support and updates, visit the BMAD community on Discord${NC}"
}

# Cleanup function
cleanup() {
    if [ "$INSTALL_METHOD" = "remote" ] && [ -n "${temp_dir:-}" ]; then
        rm -rf "$temp_dir"
    fi
}

# Error handler
error_handler() {
    local exit_code=$?
    log "ERROR" "Installation failed with exit code: $exit_code"
    log "INFO" "Check installation log: $INSTALL_LOG"
    cleanup
    exit $exit_code
}

# Main installation process
main() {
    # Handle command line arguments
    case "${1:-install}" in
        "install"|"--install"|"-i"|"")
            # Continue with installation
            ;;
        "uninstall"|"--uninstall"|"-u")
            if [ -f "/usr/local/bin/bmad-doc-automation" ]; then
                /usr/local/bin/bmad-doc-automation uninstall
            elif [ -f "$HOME/.local/bin/bmad-doc-automation" ]; then
                "$HOME/.local/bin/bmad-doc-automation" uninstall
            else
                log "ERROR" "No installation found to uninstall"
            fi
            exit 0
            ;;
        "help"|"--help"|"-h")
            print_banner
            echo -e "${BOLD}BMAD Documentation Automation Installer${NC}"
            echo ""
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  install, -i     Install the package (default)"
            echo "  uninstall, -u   Uninstall the package"
            echo "  help, -h        Show this help message"
            echo ""
            echo "Features:"
            echo "  • Enterprise documentation health monitoring"
            echo "  • Real-time status line integration"
            echo "  • Automatic link validation and fixing"
            echo "  • Documentation standards enforcement"
            echo "  • Cross-reference generation"
            echo "  • Professional documentation templates"
            echo "  • Background automation and file watching"
            echo ""
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            log "INFO" "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
    
    # Set error handler
    trap error_handler ERR
    
    # Clear log file
    > "$INSTALL_LOG"
    
    print_banner
    log "INFO" "Starting BMAD doc-automation package installation"
    
    # Installation steps
    check_requirements
    detect_installation_method
    download_package
    create_install_directory
    install_package_files
    install_bmad_integration
    install_hooks
    create_cli_interface
    
    # Test and finalize
    if test_installation; then
        show_usage_instructions
        log "SUCCESS" "BMAD Documentation Automation installed successfully!"
    else
        log "WARN" "Installation completed with some issues"
        log "INFO" "Most features should still work correctly"
    fi
    
    cleanup
}

# Execute main function
main "$@"
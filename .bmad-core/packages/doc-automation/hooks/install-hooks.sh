#!/bin/bash
# BMAD Hook Installation System
# Installs and configures BMAD documentation hooks for automatic maintenance

set -e

# Configuration
BMAD_ENABLED=${BMAD_ENABLED:-true}
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HOOK_DIR/../../../.." && pwd)"
CLAUDE_CONFIG_DIR="$HOME/.claude"
BMAD_LOG="/tmp/bmad-hook-install.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [BMAD-INSTALL] [$level] $message" >> "$BMAD_LOG"
    
    case "$level" in
        "ERROR") echo -e "${RED}❌ $message${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️ $message${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️ $message${NC}" ;;
        *) echo "$message" ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites for BMAD hook installation"
    
    local missing_deps=()
    
    # Check for bash
    if ! command -v bash > /dev/null 2>&1; then
        missing_deps+=("bash")
    fi
    
    # Check for curl (for link validation)
    if ! command -v curl > /dev/null 2>&1; then
        log "WARN" "curl not found - external link validation will be disabled"
    fi
    
    # Check for file system watchers
    if ! command -v fswatch > /dev/null 2>&1 && ! command -v inotifywait > /dev/null 2>&1; then
        log "WARN" "No file system watcher found (fswatch or inotify-tools recommended)"
        log "INFO" "Install with: brew install fswatch (macOS) or apt-get install inotify-tools (Linux)"
    fi
    
    # Check Claude Code installation
    if [ ! -d "$CLAUDE_CONFIG_DIR" ]; then
        log "WARN" "Claude Code config directory not found at $CLAUDE_CONFIG_DIR"
        log "INFO" "Some features may not work without Claude Code"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "ERROR" "Missing required dependencies: ${missing_deps[*]}"
        return 1
    fi
    
    log "SUCCESS" "Prerequisites check completed"
    return 0
}

# Install status line integration
install_status_line() {
    log "INFO" "Installing status line integration"
    
    local status_script="$HOOK_DIR/doc-health-status-line.sh"
    
    if [ ! -f "$status_script" ]; then
        log "ERROR" "Status line script not found: $status_script"
        return 1
    fi
    
    # Make script executable
    chmod +x "$status_script"
    
    # Create symlink in PATH (optional)
    local bin_dir="/usr/local/bin"
    if [ -w "$bin_dir" ] 2>/dev/null; then
        ln -sf "$status_script" "$bin_dir/bmad-doc-health"
        log "SUCCESS" "Created symlink: $bin_dir/bmad-doc-health"
    fi
    
    # Install as daemon service (systemd/launchd)
    install_daemon_service
    
    # Test status line
    if "$status_script" --test > /dev/null 2>&1; then
        log "SUCCESS" "Status line integration installed successfully"
    else
        log "ERROR" "Status line integration test failed"
        return 1
    fi
    
    return 0
}

# Install daemon service
install_daemon_service() {
    log "INFO" "Installing daemon service for automatic health monitoring"
    
    local service_name="bmad-doc-health"
    local status_script="$HOOK_DIR/doc-health-status-line.sh"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - Install LaunchAgent
        install_launchd_service "$service_name" "$status_script"
    elif command -v systemctl > /dev/null 2>&1; then
        # Linux with systemd
        install_systemd_service "$service_name" "$status_script"
    else
        # Fallback to cron
        install_cron_service "$status_script"
    fi
}

# Install macOS LaunchAgent
install_launchd_service() {
    local service_name="$1"
    local script_path="$2"
    local plist_path="$HOME/Library/LaunchAgents/com.bmad.$service_name.plist"
    
    log "INFO" "Installing macOS LaunchAgent service"
    
    cat > "$plist_path" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.bmad.$service_name</string>
    <key>ProgramArguments</key>
    <array>
        <string>$script_path</string>
        <string>--daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/tmp/bmad-doc-health-daemon.log</string>
    <key>StandardOutPath</key>
    <string>/tmp/bmad-doc-health-daemon.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>BMAD_ENABLED</key>
        <string>true</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
    
    # Load the service
    launchctl load "$plist_path" 2>/dev/null || true
    launchctl start "com.bmad.$service_name" 2>/dev/null || true
    
    log "SUCCESS" "LaunchAgent installed: $plist_path"
}

# Install Linux systemd service
install_systemd_service() {
    local service_name="$1"
    local script_path="$2"
    local service_path="$HOME/.config/systemd/user/$service_name.service"
    
    log "INFO" "Installing systemd user service"
    
    # Create user systemd directory
    mkdir -p "$(dirname "$service_path")"
    
    cat > "$service_path" << EOF
[Unit]
Description=BMAD Documentation Health Monitor
After=network.target

[Service]
Type=simple
ExecStart=$script_path --daemon
WorkingDirectory=$PROJECT_ROOT
Restart=always
RestartSec=10
Environment=BMAD_ENABLED=true
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF
    
    # Enable and start the service
    systemctl --user daemon-reload
    systemctl --user enable "$service_name.service"
    systemctl --user start "$service_name.service"
    
    log "SUCCESS" "Systemd service installed: $service_path"
}

# Install cron-based service (fallback)
install_cron_service() {
    local script_path="$1"
    
    log "INFO" "Installing cron-based service (fallback)"
    
    # Add cron job for periodic updates
    local cron_entry="*/5 * * * * cd $PROJECT_ROOT && $script_path --update-cache > /dev/null 2>&1"
    
    # Check if entry already exists
    if crontab -l 2>/dev/null | grep -F "$script_path" > /dev/null; then
        log "INFO" "Cron job already exists"
    else
        # Add to crontab
        (crontab -l 2>/dev/null || echo ""; echo "$cron_entry") | crontab -
        log "SUCCESS" "Added cron job for periodic health updates"
    fi
}

# Install post-edit hook
install_post_edit_hook() {
    log "INFO" "Installing post-edit hook"
    
    local hook_script="$HOOK_DIR/post-edit-hook.sh"
    
    if [ ! -f "$hook_script" ]; then
        log "ERROR" "Post-edit hook script not found: $hook_script"
        return 1
    fi
    
    # Make script executable
    chmod +x "$hook_script"
    
    # Install Claude Code integration if available
    if [ -d "$CLAUDE_CONFIG_DIR" ]; then
        install_claude_code_integration "$hook_script"
    fi
    
    # Create wrapper script for easy manual use
    create_hook_wrapper "$hook_script"
    
    log "SUCCESS" "Post-edit hook installed successfully"
    return 0
}

# Install Claude Code integration
install_claude_code_integration() {
    local hook_script="$1"
    local claude_hooks_dir="$CLAUDE_CONFIG_DIR/hooks"
    
    log "INFO" "Installing Claude Code integration"
    
    # Create hooks directory if it doesn't exist
    mkdir -p "$claude_hooks_dir"
    
    # Create Claude Code post-edit hook
    cat > "$claude_hooks_dir/post-edit" << EOF
#!/bin/bash
# BMAD Documentation Health Hook for Claude Code
# Automatically triggered after file edits

# Get the edited file from Claude Code environment
EDITED_FILE="\${CLAUDE_EDITED_FILE:-\$1}"

if [ -n "\$EDITED_FILE" ]; then
    # Run BMAD post-edit hook
    "$hook_script" "\$EDITED_FILE" "edit" 2>/dev/null
fi
EOF
    
    chmod +x "$claude_hooks_dir/post-edit"
    log "SUCCESS" "Claude Code post-edit hook installed"
    
    # Create settings integration
    install_claude_settings_integration
}

# Install Claude Code settings integration
install_claude_settings_integration() {
    local settings_file="$CLAUDE_CONFIG_DIR/settings.json"
    local status_script="$HOOK_DIR/doc-health-status-line.sh"
    
    log "INFO" "Installing Claude Code settings integration"
    
    # Check if settings file exists and is writable
    if [ -f "$settings_file" ] && [ -w "$settings_file" ]; then
        # Create backup
        cp "$settings_file" "$settings_file.bmad-backup"
        
        # Add status line configuration (if not already present)
        if ! grep -q "bmad-doc-health" "$settings_file" 2>/dev/null; then
            # Create temporary JSON modification
            python3 -c "
import json
import sys

try:
    with open('$settings_file', 'r') as f:
        settings = json.load(f)
except:
    settings = {}

# Add status line configuration
if 'statusLine' not in settings:
    settings['statusLine'] = {}

if 'components' not in settings['statusLine']:
    settings['statusLine']['components'] = []

# Add BMAD doc health component
doc_health_component = {
    'name': 'bmad-doc-health',
    'command': '$status_script --status-short',
    'interval': 30,
    'format': 'text'
}

# Check if component already exists
exists = any(comp.get('name') == 'bmad-doc-health' for comp in settings['statusLine']['components'])

if not exists:
    settings['statusLine']['components'].append(doc_health_component)
    
    with open('$settings_file', 'w') as f:
        json.dump(settings, f, indent=2)
    
    print('SUCCESS: Added BMAD doc health to Claude Code status line')
else:
    print('INFO: BMAD doc health already configured in status line')

" || log "WARN" "Could not automatically configure Claude Code settings (Python required)"
        else
            log "INFO" "BMAD doc health already configured in Claude Code"
        fi
    else
        log "WARN" "Could not configure Claude Code settings (file not found or not writable)"
        log "INFO" "Manual configuration: Add status line component calling: $status_script --status-short"
    fi
}

# Create hook wrapper for manual use
create_hook_wrapper() {
    local hook_script="$1"
    local wrapper_path="$PROJECT_ROOT/bmad-doc-hook"
    
    cat > "$wrapper_path" << EOF
#!/bin/bash
# BMAD Documentation Hook Wrapper
# Easy manual execution of post-edit hooks

HOOK_SCRIPT="$hook_script"
PROJECT_ROOT="$PROJECT_ROOT"

cd "\$PROJECT_ROOT"

if [ \$# -eq 0 ]; then
    echo "Usage: \$0 <file> [event_type]"
    echo ""
    echo "Examples:"
    echo "  \$0 README.md                    # Process edited file"
    echo "  \$0 docs/new-doc.md create       # Process created file"
    echo "  \$0 docs/old-doc.md delete       # Process deleted file"
    echo ""
    echo "Environment Variables:"
    echo "  BMAD_AUTO_FIX=true              # Enable automatic fixes"
    echo "  BMAD_VALIDATE_ON_EDIT=true      # Validate links on edit"
    echo "  BMAD_DEBUG=true                 # Enable debug output"
    exit 1
fi

"\$HOOK_SCRIPT" "\$@"
EOF
    
    chmod +x "$wrapper_path"
    log "SUCCESS" "Created hook wrapper: $wrapper_path"
}

# Uninstall hooks
uninstall_hooks() {
    log "INFO" "Uninstalling BMAD documentation hooks"
    
    # Stop and remove daemon services
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS LaunchAgent
        local plist_path="$HOME/Library/LaunchAgents/com.bmad.bmad-doc-health.plist"
        if [ -f "$plist_path" ]; then
            launchctl unload "$plist_path" 2>/dev/null || true
            rm -f "$plist_path"
            log "SUCCESS" "Removed LaunchAgent"
        fi
    elif command -v systemctl > /dev/null 2>&1; then
        # Linux systemd
        systemctl --user stop bmad-doc-health.service 2>/dev/null || true
        systemctl --user disable bmad-doc-health.service 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/bmad-doc-health.service"
        systemctl --user daemon-reload
        log "SUCCESS" "Removed systemd service"
    fi
    
    # Remove cron job
    if crontab -l 2>/dev/null | grep -F "doc-health-status-line.sh" > /dev/null; then
        crontab -l 2>/dev/null | grep -v "doc-health-status-line.sh" | crontab -
        log "SUCCESS" "Removed cron job"
    fi
    
    # Remove Claude Code integration
    local claude_hook="$CLAUDE_CONFIG_DIR/hooks/post-edit"
    if [ -f "$claude_hook" ]; then
        rm -f "$claude_hook"
        log "SUCCESS" "Removed Claude Code post-edit hook"
    fi
    
    # Remove symlinks
    if [ -L "/usr/local/bin/bmad-doc-health" ]; then
        rm -f "/usr/local/bin/bmad-doc-health"
        log "SUCCESS" "Removed symlink from /usr/local/bin"
    fi
    
    # Remove wrapper
    if [ -f "$PROJECT_ROOT/bmad-doc-hook" ]; then
        rm -f "$PROJECT_ROOT/bmad-doc-hook"
        log "SUCCESS" "Removed hook wrapper"
    fi
    
    # Clean up cache and log files
    rm -f /tmp/bmad-*
    
    log "SUCCESS" "BMAD hooks uninstalled successfully"
}

# Show installation status
show_status() {
    echo -e "${BLUE}BMAD Documentation Hooks Status${NC}"
    echo "=================================="
    echo ""
    
    # Check status line script
    local status_script="$HOOK_DIR/doc-health-status-line.sh"
    if [ -x "$status_script" ]; then
        echo -e "${GREEN}✅ Status Line Script${NC}: $status_script"
        if "$status_script" --test > /dev/null 2>&1; then
            echo -e "   ${GREEN}Status${NC}: Working"
        else
            echo -e "   ${RED}Status${NC}: Error"
        fi
    else
        echo -e "${RED}❌ Status Line Script${NC}: Not found or not executable"
    fi
    
    # Check post-edit hook
    local hook_script="$HOOK_DIR/post-edit-hook.sh"
    if [ -x "$hook_script" ]; then
        echo -e "${GREEN}✅ Post-Edit Hook${NC}: $hook_script"
    else
        echo -e "${RED}❌ Post-Edit Hook${NC}: Not found or not executable"
    fi
    
    # Check daemon service
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if launchctl list | grep -q "com.bmad.bmad-doc-health"; then
            echo -e "${GREEN}✅ Daemon Service${NC}: LaunchAgent running"
        else
            echo -e "${YELLOW}⚠️ Daemon Service${NC}: LaunchAgent not running"
        fi
    elif command -v systemctl > /dev/null 2>&1; then
        if systemctl --user is-active bmad-doc-health.service > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Daemon Service${NC}: systemd service running"
        else
            echo -e "${YELLOW}⚠️ Daemon Service${NC}: systemd service not running"
        fi
    else
        if crontab -l 2>/dev/null | grep -q "doc-health-status-line.sh"; then
            echo -e "${GREEN}✅ Cron Service${NC}: Periodic updates enabled"
        else
            echo -e "${YELLOW}⚠️ Cron Service${NC}: Not configured"
        fi
    fi
    
    # Check Claude Code integration
    if [ -f "$CLAUDE_CONFIG_DIR/hooks/post-edit" ]; then
        echo -e "${GREEN}✅ Claude Code Integration${NC}: Post-edit hook installed"
    else
        echo -e "${YELLOW}⚠️ Claude Code Integration${NC}: Not installed"
    fi
    
    # Check current health
    if [ -x "$status_script" ]; then
        echo ""
        echo -e "${BLUE}Current Documentation Health${NC}:"
        "$status_script" --status-colored
    fi
}

# Main execution
main() {
    case "${1:-install}" in
        "install"|"--install"|"-i")
            echo -e "${BLUE}Installing BMAD Documentation Hooks${NC}"
            echo "====================================="
            
            if ! check_prerequisites; then
                exit 1
            fi
            
            install_status_line
            install_post_edit_hook
            
            echo ""
            log "SUCCESS" "BMAD documentation hooks installed successfully!"
            log "INFO" "Status line will update automatically"
            log "INFO" "Files will be processed on edit"
            echo ""
            echo -e "${BLUE}Next Steps:${NC}"
            echo "1. Restart Claude Code to see status line updates"
            echo "2. Edit any .md file to trigger automatic processing"
            echo "3. Run: $0 --status to check installation"
            ;;
        "uninstall"|"--uninstall"|"-u")
            uninstall_hooks
            ;;
        "status"|"--status"|"-s")
            show_status
            ;;
        "test"|"--test"|"-t")
            echo -e "${BLUE}Testing BMAD Hook Installation${NC}"
            echo "==============================="
            
            check_prerequisites
            show_status
            
            # Test with a dummy file
            local test_file="/tmp/bmad-test.md"
            echo "# Test File" > "$test_file"
            echo "This is a test file for BMAD hooks." >> "$test_file"
            
            if [ -x "$HOOK_DIR/post-edit-hook.sh" ]; then
                echo ""
                echo -e "${BLUE}Testing post-edit hook with: $test_file${NC}"
                "$HOOK_DIR/post-edit-hook.sh" "$test_file" "test"
            fi
            
            rm -f "$test_file"
            ;;
        "help"|"--help"|"-h"|*)
            echo -e "${BLUE}BMAD Documentation Hooks Installer${NC}"
            echo "==================================="
            echo ""
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  install, -i     Install all BMAD documentation hooks"
            echo "  uninstall, -u   Uninstall all BMAD hooks"
            echo "  status, -s      Show installation status"
            echo "  test, -t        Test hook installation"
            echo "  help, -h        Show this help message"
            echo ""
            echo "Features Installed:"
            echo "  • Real-time status line integration with Claude Code"
            echo "  • Automatic file processing on edits"
            echo "  • Background daemon for health monitoring"
            echo "  • Link validation and auto-fixes"
            echo "  • Health score updates and caching"
            echo ""
            echo "Environment Variables:"
            echo "  BMAD_ENABLED=true           Enable BMAD integration"
            echo "  BMAD_AUTO_FIX=true          Enable automatic fixes"
            echo "  BMAD_VALIDATE_ON_EDIT=true  Validate links on edit"
            echo "  BMAD_DEBUG=true             Enable debug output"
            ;;
    esac
}

# Execute main function
main "$@"
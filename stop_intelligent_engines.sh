#!/bin/bash

# 🛑 Stop Nautilus Intelligent Engine System Script
# Gracefully stops all engine awareness components

echo "🛑 Stopping Nautilus Intelligent Engine System..."
echo "================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to stop process by PID file
stop_engine() {
    local engine_name="$1"
    local pid_file="logs/${engine_name}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat "${pid_file}")
        if ps -p $pid > /dev/null 2>&1; then
            print_info "Stopping ${engine_name} (PID: ${pid})..."
            kill -TERM $pid
            
            # Wait for graceful shutdown
            sleep 3
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "Force stopping ${engine_name}..."
                kill -KILL $pid
            fi
            
            print_status "${engine_name} stopped"
        else
            print_info "${engine_name} was not running"
        fi
        
        rm -f "${pid_file}"
    else
        print_info "No PID file found for ${engine_name}"
    fi
}

# Stop individual engines
print_info "Stopping individual engines..."

stop_engine "Enhanced ML Engine"
stop_engine "engine_coordinator"

# Kill any remaining Python processes related to our system
print_info "Cleaning up any remaining engine processes..."

# Find and stop any remaining engine processes
pkill -f "engine_coordinator.py" 2>/dev/null && print_status "Stopped remaining coordinator processes"
pkill -f "enhanced_ml_engine_with_awareness.py" 2>/dev/null && print_status "Stopped remaining ML engine processes"

# Stop infrastructure services if using Docker
if command -v docker-compose &> /dev/null; then
    print_info "Stopping infrastructure services..."
    
    docker-compose -f docker-compose.yml \
                   -f backend/docker-compose.marketdata-bus.yml \
                   -f backend/docker-compose.engine-logic-bus.yml \
                   -f backend/docker-compose.neural-gpu-bus.yml \
                   down --remove-orphans
    
    print_status "Infrastructure services stopped"
else
    print_warning "Docker Compose not found - infrastructure services may still be running"
fi

# Clean up log files (optional - comment out if you want to keep logs)
print_info "Cleaning up temporary files..."
rm -f logs/*.pid

print_info "Log files preserved in logs/ directory"

# Final check
print_info "Checking for any remaining processes..."
remaining=$(pgrep -f "(engine_coordinator|enhanced_ml_engine)" | wc -l)

if [ $remaining -eq 0 ]; then
    print_status "All engine processes stopped successfully"
else
    print_warning "${remaining} processes may still be running"
    print_info "You can check with: ps aux | grep -E '(engine_coordinator|enhanced_ml_engine)'"
fi

# Show final status
echo ""
echo "🔍 Final System Status:"
echo "======================="

if curl -s --max-time 2 http://localhost:8000/health > /dev/null 2>&1; then
    print_warning "Engine Coordinator still responding (may take a moment to shut down)"
else
    print_status "Engine Coordinator stopped"
fi

if curl -s --max-time 2 http://localhost:8400/health > /dev/null 2>&1; then
    print_warning "Enhanced ML Engine still responding"
else
    print_status "Enhanced ML Engine stopped"
fi

echo ""
echo "📋 To check system status:"
echo "========================="
echo "View remaining processes: ps aux | grep python"
echo "Check port usage:        lsof -i :8000,8400"
echo "View logs:               ls -la logs/"

echo ""
print_status "🛑 Nautilus Intelligent Engine System shutdown complete!"

echo ""
print_info "To restart the system, run: ./start_intelligent_engines.sh"
#!/bin/bash

# 🧠 Nautilus Intelligent Engine System Startup Script
# Starts the Engine Interconnection & Awareness System

echo "🧠 Starting Nautilus Intelligent Engine System..."
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the correct directory
if [ ! -f "backend/engine_coordinator.py" ]; then
    print_error "Please run this script from the Nautilus root directory"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PWD}/backend:${PYTHONPATH}"

print_info "Setting up Python path: ${PYTHONPATH}"

# Start infrastructure services (Redis, PostgreSQL)
print_info "Starting infrastructure services..."

if command -v docker-compose &> /dev/null; then
    print_status "Starting Redis message buses..."
    docker-compose -f docker-compose.yml \
                   -f backend/docker-compose.marketdata-bus.yml \
                   -f backend/docker-compose.engine-logic-bus.yml \
                   -f backend/docker-compose.neural-gpu-bus.yml \
                   up -d postgres \
                      marketdata-redis-cluster \
                      engine-logic-redis-cluster \
                      neural-gpu-redis-cluster \
                      prometheus grafana
    
    print_status "Infrastructure services started"
else
    print_warning "Docker Compose not found - you may need to start Redis and PostgreSQL manually"
fi

# Wait for services to be ready
print_info "Waiting for infrastructure services to be ready..."
sleep 10

# Function to start engine in background
start_engine() {
    local engine_name="$1"
    local engine_script="$2"
    local port="$3"
    
    print_info "Starting ${engine_name}..."
    
    if [ -f "${engine_script}" ]; then
        cd backend
        python "${engine_script}" > "../logs/${engine_name}.log" 2>&1 &
        engine_pid=$!
        echo $engine_pid > "../logs/${engine_name}.pid"
        cd ..
        
        print_status "${engine_name} started (PID: ${engine_pid}, Port: ${port})"
        
        # Give the engine a moment to start
        sleep 2
    else
        print_warning "${engine_name} script not found: ${engine_script}"
    fi
}

# Create logs directory
mkdir -p logs

print_info "Starting Engine Coordinator (Central Intelligence)..."

# Start the Engine Coordinator first
cd backend
python engine_coordinator.py > ../logs/engine_coordinator.log 2>&1 &
coordinator_pid=$!
echo $coordinator_pid > ../logs/engine_coordinator.pid
cd ..

print_status "Engine Coordinator started (PID: ${coordinator_pid}, Port: 8000)"

# Wait for coordinator to be ready
print_info "Waiting for Engine Coordinator to initialize..."
sleep 15

# Start enhanced engines
print_info "Starting enhanced engines with awareness capabilities..."

# Enhanced ML Engine with full awareness
start_engine "Enhanced ML Engine" "engines/examples/enhanced_ml_engine_with_awareness.py" "8400"

# Give engines time to discover each other
print_info "Allowing engines to discover each other..."
sleep 10

# Check system status
print_info "Checking system status..."

# Test coordinator health
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "Engine Coordinator is responding"
else
    print_error "Engine Coordinator is not responding"
fi

# Test enhanced ML engine
if curl -s http://localhost:8400/health > /dev/null; then
    print_status "Enhanced ML Engine is responding"
else
    print_warning "Enhanced ML Engine is not responding"
fi

# Display system information
echo ""
echo "🎯 System Status:"
echo "=================="

print_info "Engine Coordinator Dashboard: http://localhost:8000/docs"
print_info "System Status API: http://localhost:8000/api/v1/system/status"
print_info "AI Agents Status: http://localhost:8000/api/v1/ai-agents/status"
print_info "Network Topology: http://localhost:8000/api/v1/network/topology"
print_info "System Intelligence: http://localhost:8000/api/v1/system/intelligence"

if command -v docker-compose &> /dev/null; then
    print_info "Grafana Dashboard: http://localhost:3002"
    print_info "Prometheus Metrics: http://localhost:9090"
fi

print_info "Enhanced ML Engine: http://localhost:8400/docs"

echo ""
echo "📊 Monitoring Commands:"
echo "======================="
echo "View system status:    curl http://localhost:8000/api/v1/system/status | python3 -m json.tool"
echo "View AI decisions:     curl -X POST http://localhost:8000/api/v1/ai-agents/decision | python3 -m json.tool"  
echo "View partnerships:     curl http://localhost:8000/api/v1/partnerships | python3 -m json.tool"
echo "View engine network:   curl http://localhost:8000/api/v1/network/topology | python3 -m json.tool"

echo ""
echo "🔧 Management Commands:"
echo "======================="
echo "Stop all engines:      ./stop_intelligent_engines.sh"
echo "View engine logs:      tail -f logs/*.log"
echo "Check process status:  ps aux | grep python"

echo ""
echo "🧠 AI Agent Commands:"
echo "====================="
echo "Get AI decision:       curl -X POST http://localhost:8000/api/v1/ai-agents/decision"
echo "Create strategy:       curl -X POST http://localhost:8000/api/v1/ai-agents/strategy -d '[\"optimize_trading\", \"improve_performance\"]' -H 'Content-Type: application/json'"
echo "AI decision history:   curl http://localhost:8000/api/v1/ai-agents/decisions/history"

echo ""
print_status "🚀 Nautilus Intelligent Engine System is running!"
print_info "Watch the logs to see engines discovering each other and forming partnerships:"
echo ""
echo "tail -f logs/engine_coordinator.log"
echo "tail -f logs/Enhanced\\ ML\\ Engine.log"

echo ""
print_info "The AI agents will start making decisions in ~2 minutes..."
print_info "System intelligence will grow as engines collaborate with real data!"

echo ""
echo "🎭 Expected Behavior:"
echo "===================="
print_info "1. Engines will announce themselves and discover each other"
print_info "2. Partnerships will form based on capabilities and preferences" 
print_info "3. AI agents will analyze the system and make optimization recommendations"
print_info "4. Performance metrics will improve as the system learns"
print_info "5. New collaboration patterns will emerge over time"

echo ""
print_status "System startup complete! 🎉"
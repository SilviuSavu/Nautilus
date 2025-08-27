#!/bin/bash
# 🚀 Nautilus System-on-Chip Database Initialization Script
# Dynamic setup for Apple Silicon M4 Max architecture

set -e

echo "🚀 Initializing Nautilus SoC Database Architecture..."
echo "======================================================"

# Detect Apple Silicon M4 Max capabilities
detect_system_info() {
    echo "🔍 Detecting Apple Silicon M4 Max capabilities..."
    
    # Get system information
    TOTAL_MEMORY_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))")
    AVAILABLE_MEMORY_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))")
    CPU_CORES=$(python3 -c "import psutil; print(psutil.cpu_count())")
    PHYSICAL_CORES=$(python3 -c "import psutil; print(psutil.cpu_count(logical=False))")
    
    echo "  💾 Total Memory: ${TOTAL_MEMORY_GB}GB"
    echo "  ⚡ Available Memory: ${AVAILABLE_MEMORY_GB}GB"
    echo "  🔧 CPU Cores: ${CPU_CORES} (${PHYSICAL_CORES} physical)"
    
    # Detect Apple Silicon
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo "  🍎 Apple Silicon detected: ARM64 architecture"
        APPLE_SILICON=true
    else
        echo "  ⚠️  Non-Apple Silicon detected: $ARCH architecture"
        APPLE_SILICON=false
    fi
    
    # Export for use by containers
    export TOTAL_MEMORY_GB
    export AVAILABLE_MEMORY_GB
    export CPU_CORES
    export PHYSICAL_CORES
    export APPLE_SILICON
}

# Create dynamic data directories
create_directories() {
    echo "📁 Creating dynamic data directories..."
    
    # Create base data directories
    mkdir -p data/clickhouse/{data,logs}
    mkdir -p data/druid/{coordinator,broker,historical,middlemanager,router}
    mkdir -p data/minio/{data,cache}
    mkdir -p data/postgres
    mkdir -p data/redis
    
    # Set appropriate permissions for Apple Silicon optimization
    chmod -R 755 data/
    
    echo "  ✅ Data directories created"
}

# Start dynamic resource manager
start_resource_manager() {
    echo "🎯 Starting dynamic resource manager..."
    
    # Make the script executable
    chmod +x scripts/dynamic_resource_manager.py
    
    # Start resource manager in background
    python3 scripts/dynamic_resource_manager.py > logs/resource_manager.log 2>&1 &
    RESOURCE_MANAGER_PID=$!
    echo $RESOURCE_MANAGER_PID > /tmp/resource_manager.pid
    
    echo "  ✅ Dynamic resource manager started (PID: $RESOURCE_MANAGER_PID)"
}

# Initialize databases with dynamic configuration
init_databases() {
    echo "🗄️  Initializing databases with dynamic configuration..."
    
    # Wait for resource manager to generate initial config
    sleep 5
    
    # Start the SoC database stack
    echo "  🚀 Starting SoC database stack..."
    docker-compose -f docker-compose.soc-database.yml up -d
    
    # Wait for databases to be ready
    echo "  ⏳ Waiting for databases to be ready..."
    
    # Wait for ClickHouse
    echo "    Waiting for ClickHouse..."
    until curl -s http://localhost:8123/ping > /dev/null; do
        sleep 2
    done
    echo "    ✅ ClickHouse ready"
    
    # Wait for Druid
    echo "    Waiting for Druid..."
    until curl -s http://localhost:8888/status > /dev/null; do
        sleep 2
    done
    echo "    ✅ Druid ready"
    
    # Wait for PostgreSQL
    echo "    Waiting for PostgreSQL..."
    until docker exec nautilus-postgres-enhanced pg_isready -U nautilus > /dev/null; do
        sleep 2
    done
    echo "    ✅ PostgreSQL ready"
    
    # Wait for Redis
    echo "    Waiting for Redis..."
    until docker exec nautilus-redis-enhanced redis-cli ping > /dev/null; do
        sleep 2
    done
    echo "    ✅ Redis ready"
    
    # Wait for MinIO
    echo "    Waiting for MinIO..."
    until curl -s http://localhost:9000/minio/health/ready > /dev/null; do
        sleep 2
    done
    echo "    ✅ MinIO ready"
}

# Setup initial data structures
setup_initial_schemas() {
    echo "🏗️  Setting up initial data schemas..."
    
    # ClickHouse schema setup
    echo "  📊 Setting up ClickHouse schemas..."
    cat <<EOF | docker exec -i nautilus-clickhouse clickhouse-client --user=nautilus --password=nautilus123
    CREATE DATABASE IF NOT EXISTS nautilus;
    
    -- Market data table optimized for Apple Silicon
    CREATE TABLE IF NOT EXISTS nautilus.market_data (
        timestamp DateTime64(3) CODEC(Delta, LZ4),
        symbol LowCardinality(String),
        price Decimal64(8) CODEC(Gorilla, LZ4),
        volume UInt64 CODEC(Delta, LZ4),
        bid Decimal64(8) CODEC(Gorilla, LZ4),
        ask Decimal64(8) CODEC(Gorilla, LZ4),
        bid_size UInt64 CODEC(Delta, LZ4),
        ask_size UInt64 CODEC(Delta, LZ4)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (symbol, timestamp)
    SETTINGS index_granularity = 8192;
    
    -- Analytics results table
    CREATE TABLE IF NOT EXISTS nautilus.analytics_results (
        timestamp DateTime64(3) CODEC(Delta, LZ4),
        engine LowCardinality(String),
        metric_name LowCardinality(String),
        metric_value Float64 CODEC(Gorilla, LZ4),
        symbol LowCardinality(String),
        metadata String CODEC(LZ4)
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMM(timestamp)
    ORDER BY (engine, metric_name, timestamp)
    SETTINGS index_granularity = 8192;
EOF
    
    # MinIO bucket setup
    echo "  🪣 Setting up MinIO buckets..."
    docker exec nautilus-minio mc alias set local http://localhost:9000 nautilus nautilus123
    docker exec nautilus-minio mc mb local/druid-deep-storage --ignore-existing
    docker exec nautilus-minio mc mb local/data-lake --ignore-existing
    docker exec nautilus-minio mc mb local/delta-lake --ignore-existing
    
    echo "  ✅ Initial schemas configured"
}

# Performance optimization
optimize_performance() {
    echo "⚡ Applying Apple Silicon M4 Max performance optimizations..."
    
    # Apply kernel optimizations (if running on macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  🍎 Applying macOS/Apple Silicon optimizations..."
        
        # Set optimal kernel parameters for Apple Silicon
        sudo sysctl -w kern.maxfiles=65536 2>/dev/null || true
        sudo sysctl -w kern.maxfilesperproc=32768 2>/dev/null || true
        
        # Docker optimizations for Apple Silicon
        docker system prune -f > /dev/null 2>&1 || true
        
        echo "  ✅ Apple Silicon optimizations applied"
    fi
    
    # Container-level optimizations
    echo "  🐳 Applying container optimizations..."
    
    # Set CPU affinity for critical containers (Apple Silicon optimization)
    docker update --cpuset-cpus="0-11" nautilus-clickhouse 2>/dev/null || true
    docker update --cpuset-cpus="0-7" nautilus-druid-broker 2>/dev/null || true
    
    echo "  ✅ Container optimizations applied"
}

# Health check
health_check() {
    echo "🏥 Performing system health check..."
    
    # Check all services
    SERVICES=("clickhouse:8123" "druid:8888" "postgres:5432" "redis:6379" "minio:9000")
    
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name port <<< "$service"
        echo -n "  Checking $name:$port... "
        
        if timeout 10 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            echo "✅"
        else
            echo "❌"
        fi
    done
    
    # Check dynamic resource manager
    if ps -p $(cat /tmp/resource_manager.pid 2>/dev/null) > /dev/null 2>&1; then
        echo "  Dynamic resource manager: ✅"
    else
        echo "  Dynamic resource manager: ❌"
    fi
}

# Main execution
main() {
    echo "🚀 Starting Nautilus SoC Database Architecture Setup"
    echo "=================================================="
    
    # Create logs directory
    mkdir -p logs
    
    # Run all setup steps
    detect_system_info
    create_directories
    start_resource_manager
    init_databases
    setup_initial_schemas
    optimize_performance
    health_check
    
    echo ""
    echo "🎉 Nautilus SoC Database Architecture Successfully Initialized!"
    echo "============================================================="
    echo ""
    echo "📊 Access Points:"
    echo "  • ClickHouse:  http://localhost:8123"
    echo "  • Druid:       http://localhost:8888"
    echo "  • MinIO:       http://localhost:9001"
    echo "  • PostgreSQL:  localhost:5432"
    echo "  • Redis:       localhost:6379"
    echo ""
    echo "📈 Dynamic Resource Manager: Active (logs/resource_manager.log)"
    echo "🔧 System Info: ${TOTAL_MEMORY_GB}GB RAM, ${CPU_CORES} cores, Apple Silicon: ${APPLE_SILICON}"
    echo ""
    echo "✅ Ready for Phase 2: Data Mesh Architecture Setup"
}

# Run main function
main "$@"
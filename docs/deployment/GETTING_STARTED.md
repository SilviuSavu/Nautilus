# Getting Started

## M4 Max Optimized Deployment (Recommended)
**PRODUCTION-READY**: ✅ **M4 Max hardware acceleration with 50x+ performance improvements**
**IMPORTANT: All services run in Docker containers. Do NOT run services locally.**
**CRITICAL: NO hardcoded values in frontend - all use environment variables.**

### Quick Start (M4 Max Optimized)
1. Ensure Docker and Docker Compose are installed
2. **M4 Max Detection**: System automatically detects Apple M4 Max hardware
3. **Hardware Acceleration Setup**:
   - Metal Performance Shaders enabled
   - Neural Engine integration active
   - 14-core CPU optimization (10 P-cores + 4 E-cores)
   - 36GB unified memory management
4. API keys are already configured in docker-compose.yml:
   - ALPHA_VANTAGE_API_KEY=271AHP91HVAPDRGP
   - FRED_API_KEY=1f1ba9c949e988e12796b7c1f6cce1bf
5. **Start M4 Max Optimized System**: `./start-m4max-optimized.sh`
   - Alternative: `docker-compose -f docker-compose.m4max.yml up`
6. **Access Points**:
   - **Backend**: http://localhost:8001 (M4 Max optimized)
   - **Frontend**: http://localhost:3000 (containerized)
   - **Database**: localhost:5432 (TimescaleDB optimized)
   - **Redis**: localhost:6379 (M4 Max memory optimized)
   - **Prometheus**: http://localhost:9090 (M4 Max monitoring)
   - **Grafana**: http://localhost:3002 (M4 Max dashboards)

### Legacy Deployment (Standard Docker)
For non-M4 Max systems, use standard deployment: `docker-compose up`

## Environment Variables (M4 Max Optimized)
- **VITE_API_BASE_URL**: http://localhost:8001 (frontend → backend)
- **VITE_WS_URL**: localhost:8001 (WebSocket connections)
- **M4_MAX_OPTIMIZATION**: true (enables hardware acceleration)
- **METAL_PERFORMANCE_SHADERS**: true (enables Metal GPU acceleration)
- **NEURAL_ENGINE**: true (enables Neural Engine for ML inference)
- **CPU_OPTIMIZATION**: m4max (14-core optimization profile)
- **UNIFIED_MEMORY**: 36gb (M4 Max memory configuration)
- **All frontend components**: Use environment variables, NO hardcoded URLs

## Data Provider Configuration (M4 Max Enhanced)
- **IBKR**: Interactive Brokers Gateway connection (M4 Max low-latency processing)
- **Alpha Vantage**: Set environment variable `ALPHA_VANTAGE_API_KEY` (Metal acceleration)
- **FRED**: Set environment variable `FRED_API_KEY` (Neural Engine macro analysis)
- **EDGAR**: No API key required (M4 Max parallel processing)
- **Data.gov**: 346,000+ datasets with M4 Max relevance scoring
- **DBnomics**: 800M+ time series with hardware-accelerated processing
- **Trading Economics**: Global economic data with M4 Max optimization
- **Yahoo Finance**: Real-time data with unified memory caching

## M4 Max Performance Features

### Hardware Acceleration
- **Metal Performance Shaders**: GPU acceleration for ML workloads
- **Neural Engine**: 16-core Neural Engine for inference (15.8 TOPS)
- **CPU Optimization**: Intelligent P-core/E-core workload distribution
- **Memory Bandwidth**: 400GB/s unified memory architecture
- **Thermal Management**: Intelligent thermal monitoring and throttling prevention

### Containerized Engine Architecture (9 Engines)
- **Analytics Engine**: 3.0 CPU cores, 6GB RAM (Metal acceleration)
- **Risk Engine**: 1.0 CPU core, 2GB RAM (ultra-low latency)
- **Factor Engine**: 4.0 CPU cores, 12GB RAM (heavy computation)
- **ML Engine**: 3.0 CPU cores, 10GB RAM (Neural Engine integration)
- **Features Engine**: 2.0 CPU cores, 6GB RAM (vector optimization)
- **WebSocket Engine**: 1.0 CPU core, 3GB RAM (real-time communication)
- **Strategy Engine**: 1.5 CPU cores, 4GB RAM (parallel execution)
- **MarketData Engine**: 2.0 CPU cores, 4GB RAM (data compression)
- **Portfolio Engine**: 3.0 CPU cores, 12GB RAM (advanced optimization)

### Performance Monitoring
- **Real-time Hardware Metrics**: M4 Max chip monitoring
- **Container Performance**: Individual engine optimization
- **Trading Performance**: Latency and throughput tracking
- **Thermal Management**: Background thermal state monitoring
- **Resource Alerts**: Intelligent alerting system

## Development Guidelines
- Follow standard coding practices for each language
- Write comprehensive tests for new functionality
- Use proper error handling and logging
- Maintain clean, readable code with good documentation
- **M4 Max Optimization**: Leverage hardware acceleration APIs where possible
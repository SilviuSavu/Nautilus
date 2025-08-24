# Docker Setup & Commands

## M4 Max Optimized Commands (Recommended)
**Production-Ready M4 Max deployment with hardware acceleration**

### M4 Max Startup Commands
- **Optimized Startup**: `./start-m4max-optimized.sh` (intelligent hardware detection)
- **Manual M4 Max**: `docker-compose -f docker-compose.m4max.yml up`
- **Background M4 Max**: `docker-compose -f docker-compose.m4max.yml up -d`
- **M4 Max with Build**: `docker-compose -f docker-compose.m4max.yml up --build`
- **View M4 Max Logs**: `docker-compose -f docker-compose.m4max.yml logs [service-name]`

## Legacy Commands (Standard Docker)
**All commands assume Docker containers are running with `docker-compose up`**

### M4 Max Container Management
- **M4 Max Health Check**: `./start-m4max-optimized.sh --health-check`
- **Stop M4 Max Services**: `docker-compose -f docker-compose.m4max.yml down`
- **M4 Max Resource Monitor**: `docker stats nautilus-analytics-engine:m4max nautilus-risk-engine:m4max`
- **M4 Max Thermal Monitor**: `tail -f thermal-monitor.log`
- **Container CPU Assignment**: View CPU affinity with `docker exec nautilus-analytics-engine:m4max cat /proc/self/status | grep Cpus_allowed_list`

### Legacy Container Management
- Start all services: `docker-compose up`
- Start in background: `docker-compose up -d`
- Stop all services: `docker-compose down`
- Rebuild containers: `docker-compose up --build`
- View logs: `docker-compose logs [service-name]`

### M4 Max Performance Testing
- **M4 Max Performance Benchmark**: `python backend/scripts/test_m4_max_performance.py`
- **Hardware Validation**: `python backend/benchmarks/hardware_validation.py`
- **Container Performance**: `python backend/benchmarks/container_benchmarks.py`
- **Neural Engine Test**: `python backend/engines/ml/neural_engine_test.py`
- **Metal GPU Benchmark**: `python backend/engines/ml/metal_gpu_benchmark.py`
- **Unified Memory Test**: `python backend/memory/unified_memory_test.py`
- **CPU Optimization Test**: `python backend/optimization/test_cpu_optimization.py`

### Legacy Testing
- Run frontend tests: `cd frontend && npm test`
- Run backend tests: `docker exec nautilus-backend pytest`
- Run Playwright tests: `cd frontend && npx playwright test`
- Run Playwright headed: `cd frontend && npx playwright test --headed`

#### Sprint 3 Testing (NEW)
- **Run all Sprint 3 tests**: `pytest backend/tests/ -v`
- **WebSocket load testing**: `pytest backend/tests/test_websocket_scalability.py`
- **Integration testing**: `pytest -m integration backend/tests/`
- **Performance benchmarks**: `pytest backend/tests/test_performance_benchmarks.py`
- **Risk management tests**: `pytest -m risk backend/tests/`
- **Strategy framework tests**: `pytest -m strategy backend/tests/`

### M4 Max Engine Health Checks (Hardware Accelerated)
- **M4 Max System**: `curl http://localhost:8001/health` (hardware acceleration status)
- **Analytics Engine**: `curl http://localhost:8100/health` (Metal Performance Shaders)
- **Risk Engine**: `curl http://localhost:8200/health` (ultra-low latency)
- **Factor Engine**: `curl http://localhost:8300/health` (heavy computation optimized)
- **ML Engine**: `curl http://localhost:8400/health` (Neural Engine integration)
- **Features Engine**: `curl http://localhost:8500/health` (vector optimization)
- **WebSocket Engine**: `curl http://localhost:8600/health` (real-time communication)
- **Strategy Engine**: `curl http://localhost:8700/health` (parallel execution)
- **MarketData Engine**: `curl http://localhost:8800/health` (data compression)
- **Portfolio Engine**: `curl http://localhost:8900/health` (advanced optimization)

### M4 Max Hardware Monitoring
- **Hardware Metrics**: `curl http://localhost:8001/api/v1/monitoring/m4max/hardware/metrics`
- **Container Metrics**: `curl http://localhost:8001/api/v1/monitoring/containers/metrics`
- **Trading Performance**: `curl http://localhost:8001/api/v1/monitoring/trading/metrics`
- **System Health**: `curl http://localhost:8001/api/v1/monitoring/system/health`
- **Active Alerts**: `curl http://localhost:8001/api/v1/monitoring/alerts`

### Legacy Health Checks (All Containerized)
- System: `curl http://localhost:8001/health`
- Unified data sources: `curl http://localhost:8001/api/v1/nautilus-data/health`
- FRED macro factors: `curl http://localhost:8001/api/v1/nautilus-data/fred/macro-factors`
- Alpha Vantage search: `curl "http://localhost:8001/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL"`
- EDGAR: `curl http://localhost:8001/api/v1/edgar/health`

### M4 Max Optimization APIs (NEW)
- **CPU Optimization Health**: `curl http://localhost:8001/api/v1/optimization/health`
- **M4 Max Core Utilization**: `curl http://localhost:8001/api/v1/optimization/core-utilization`
- **Performance Optimization**: `curl http://localhost:8001/api/v1/monitoring/performance/optimizations`
- **Hardware Acceleration Status**: `curl http://localhost:8001/api/v1/optimization/system-info`
- **Container Performance**: `curl http://localhost:8001/api/v1/optimization/containers/stats`

#### Legacy Sprint 3 Health Checks
- **WebSocket infrastructure**: `curl http://localhost:8001/api/v1/websocket/health`
- **Risk management**: `curl http://localhost:8001/api/v1/risk/health`
- **Analytics pipeline**: `curl http://localhost:8001/api/v1/analytics/health`
- **Strategy framework**: `curl http://localhost:8001/api/v1/strategies/health`
- **System monitoring**: `curl http://localhost:8001/api/v1/system/health`

### Database Access
- PostgreSQL: `docker exec -it nautilus-postgres psql -U nautilus -d nautilus`
- Redis: `docker exec -it nautilus-redis redis-cli`

### M4 Max Engine Management (Hardware Optimized)
- **Start M4 Max Engine**: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/start -H "Content-Type: application/json" -d '{"config": {"engine_type": "live", "log_level": "INFO", "instance_id": "m4max-001", "trading_mode": "paper", "max_memory": "12g", "max_cpu": "4.0", "hardware_acceleration": true, "metal_performance_shaders": true, "neural_engine": true, "cpu_affinity": "performance_cores", "data_catalog_path": "/app/data", "cache_database_path": "/app/cache", "risk_engine_enabled": true}, "confirm_live_trading": false}'`
- **M4 Max Engine Status**: `curl http://localhost:8001/api/v1/nautilus/engine/status` (includes hardware metrics)
- **Stop M4 Max Engine**: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/stop -H "Content-Type: application/json" -d '{"force": false, "preserve_hardware_state": true}'`
- **M4 Max Performance Logs**: `curl http://localhost:8001/api/v1/nautilus/engine/logs?lines=50&include_hardware_metrics=true`
- **Force Container Optimization**: `curl -X POST http://localhost:8001/api/v1/optimization/containers/optimize`

### Legacy Engine Management
- Start engine: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/start -H "Content-Type: application/json" -d '{"config": {"engine_type": "live", "log_level": "INFO", "instance_id": "test-001", "trading_mode": "paper", "max_memory": "2g", "max_cpu": "2.0", "data_catalog_path": "/app/data", "cache_database_path": "/app/cache", "risk_engine_enabled": true}, "confirm_live_trading": false}'`
- Check status: `curl http://localhost:8001/api/v1/nautilus/engine/status`  
- Stop engine: `curl -X POST http://localhost:8001/api/v1/nautilus/engine/stop -H "Content-Type: application/json" -d '{"force": false}'`
- View logs: `curl http://localhost:8001/api/v1/nautilus/engine/logs?lines=50`
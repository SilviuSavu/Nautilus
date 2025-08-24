# Troubleshooting Guide

## 🚨 CRITICAL FIXES & TROUBLESHOOTING (Sprint 2)

### Frontend-Backend API Structure Mismatch (FIXED)
**Problem**: Frontend shows `TypeError: Cannot read properties of undefined (reading 'replace')`
**Root Cause**: Backend returned nested structures by session ID, frontend expected flat objects
**Fix Location**: `/backend/nautilus_engine_service.py` lines 446-456 in `get_engine_status()`

```python
# CRITICAL FIX: Flatten nested structures for single container scenarios
if len(self.dynamic_containers) == 1:
    session_id = list(self.dynamic_containers.keys())[0]
    if isinstance(resource_usage, dict) and session_id in resource_usage:
        resource_usage = resource_usage[session_id]
    if isinstance(container_info, dict) and session_id in container_info:
        container_info = container_info[session_id]
```

**Before (BROKEN)**:
```json
{
  "resource_usage": {
    "session-id-123": {"cpu_percent": "1.5%", "memory_usage": "100MB"}
  },
  "container_info": {
    "session-id-123": {"status": "running", "name": "nautilus-engine-xyz"}
  }
}
```

**After (FIXED)**:
```json
{
  "resource_usage": {"cpu_percent": "1.5%", "memory_usage": "100MB"},
  "container_info": {"status": "running", "name": "nautilus-engine-xyz"}
}
```

### Network Configuration Issues
- **Docker Network**: MUST use `nautilus_nautilus-network` (not `nautilus-network`)
- **CORS Origins**: Include both port 3000 and 3001 in docker-compose.yml
- **IB Gateway**: Use mocked imports if real nautilus_trading_node fails import

### Container Management Issues
- **Container Cleanup**: Use `docker container prune -f` to clean stopped containers
- **Image Updates**: `docker-compose build backend` to rebuild engine image
- **Health Checks**: Engine containers expose health endpoint on port 8001

### Port Configuration (DO NOT CHANGE)
- **Frontend**: localhost:3000 (Docker container)
- **Backend**: localhost:8001 (Docker container) 
- **Database**: localhost:5432 (Docker container)
- **Frontend Environment**: VITE_API_BASE_URL=http://localhost:8001

## M4 Max Performance Testing & Validation
- **M4 Max Benchmarking**: `python backend/scripts/test_m4_max_performance.py`
- **Hardware Validation**: `python backend/benchmarks/hardware_validation.py`
- **Container Performance**: `python backend/benchmarks/container_benchmarks.py`
- **Neural Engine Benchmarks**: `python backend/engines/ml/neural_engine_benchmark.py`
- **Metal GPU Testing**: `python backend/engines/ml/metal_gpu_benchmark.py`
- **Memory Pool Testing**: `python backend/memory/memory_pool_validation.py`
- **CPU Optimization Testing**: `python backend/optimization/test_cpu_optimization.py`
- **Production Workload Simulation**: `python backend/engines/ml/production_workload_simulation.py`
- **Comprehensive System Assessment**: `python backend/engines/ml/comprehensive_system_assessment.py`
- **Reliability Stress Testing**: `python backend/engines/ml/reliability_stress_test.py`

### Legacy Testing Framework
- **E2E Testing**: Playwright for browser automation and integration testing
- **Playwright MCP**: Available for advanced test automation via MCP protocol
- **Unit Tests**: Vitest (frontend), pytest (backend)
- **Component Tests**: React Testing Library with Vitest
- **Sprint 3 Testing**: Comprehensive test suite with 14 test files
  - **Load Testing**: 1000+ concurrent WebSocket connections validated
  - **Integration Testing**: End-to-end workflow testing for all components
  - **Performance Testing**: 50,000+ messages/second throughput benchmarks
  - **Coverage**: >85% test coverage across all Sprint 3 components

## Playwright Integration
- Playwright tests located in `frontend/tests/e2e/`
- Use Playwright for end-to-end workflow testing
- Playwright MCP server available for advanced automation scenarios
- Browser automation for real user interaction testing
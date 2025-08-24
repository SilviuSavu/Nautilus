# CPU Core Optimization System for M4 Max

## Overview

This comprehensive CPU core optimization system is specifically designed for the Apple M4 Max architecture to achieve ultra-low latency trading operations while maximizing overall system throughput. The system intelligently manages CPU core allocation, process scheduling, and workload distribution across the M4 Max's 12 performance cores and 4 efficiency cores.

## Architecture

### M4 Max Configuration
- **Performance Cores (P-cores)**: 12 cores (IDs 0-11)
  - Base: 3.228 GHz, Max: 4.050 GHz
  - High performance, optimized for single-threaded workloads
  - Assigned to trading-critical operations
  
- **Efficiency Cores (E-cores)**: 4 cores (IDs 12-15)
  - Base/Max: 2.750 GHz
  - Power efficient, suitable for background tasks
  - Assigned to maintenance and non-critical operations

## Core Components

### 1. CPU Affinity Manager (`cpu_affinity.py`)
**Purpose**: Intelligent CPU core detection and allocation for M4 Max architecture

**Key Features**:
- Automatic M4 Max architecture detection
- Real-time per-core utilization monitoring
- Core assignment based on workload priority
- Dynamic load balancing and migration
- Thermal state tracking

**Core Assignment Strategy**:
- **Ultra-Low Latency** (P-cores 0-3): Order execution, market data ticks
- **Low Latency** (P-cores 4-7): Risk calculations, position updates  
- **Normal Priority** (P-cores 8-11): Analytics, reporting
- **Background** (E-cores 12-15): Data backfill, maintenance

### 2. Process Manager (`process_manager.py`)
**Purpose**: Advanced process management with priority-based scheduling

**Key Features**:
- Process classification by trading importance
- QoS-based CPU priority assignment
- Resource limit enforcement
- Market condition-aware priority adjustment
- Automatic process lifecycle management

**Process Classes**:
- **Trading Core**: Order management, execution engines
- **Risk Management**: Risk calculations, position monitoring
- **Analytics**: Performance analysis, reporting
- **Data Processing**: Market data feeds, factor calculations
- **Background**: Maintenance, cleanup, logging

### 3. GCD Scheduler (`gcd_scheduler.py`)
**Purpose**: Native macOS Grand Central Dispatch integration

**Key Features**:
- Quality of Service (QoS) class management
- Dispatch queue optimization for trading workloads
- Thread pool management per QoS level
- Task grouping and synchronization
- Performance monitoring and statistics

**QoS Classes**:
- **User Interactive** (33): Trading execution, order management
- **User Initiated** (25): Market data, risk calculations
- **Default** (21): Analytics, standard processing
- **Utility** (17): Data processing, batch operations
- **Background** (9): Maintenance, cleanup tasks

### 4. Performance Monitor (`performance_monitor.py`)
**Purpose**: Real-time system performance tracking and alerting

**Key Features**:
- Per-core CPU utilization monitoring
- Latency measurement for trading operations
- Memory usage and thermal state tracking
- Automated alert generation and escalation
- SQLite-based metrics persistence
- Performance baseline tracking

**Monitored Metrics**:
- CPU utilization (per-core and aggregate)
- Memory usage (system and per-process)
- Operation latency (with percentile analysis)
- Thermal state and power consumption
- I/O statistics (disk and network)
- Alert counts and system health score

### 5. Workload Classifier (`workload_classifier.py`)
**Purpose**: ML-based intelligent task categorization and optimization

**Key Features**:
- Machine learning workload classification
- Feature extraction from execution context
- Heuristic rule-based fallback classification
- Automatic model retraining with new data
- Performance-based priority assignment
- Classification confidence scoring

**Workload Categories**:
- **Trading Execution**: Order placement, execution
- **Market Data**: Data ingestion, tick processing
- **Risk Calculation**: Portfolio risk, position sizing
- **Analytics**: Performance analysis, reporting
- **ML Inference**: Model predictions, analysis
- **Data Processing**: ETL, transformation
- **Background Maintenance**: Cleanup, archiving
- **System Monitoring**: Health checks, diagnostics

### 6. Optimizer Controller (`optimizer_controller.py`)
**Purpose**: Central orchestration and coordination

**Key Features**:
- Component lifecycle management
- Configuration-driven optimization policies
- Automatic mode switching (market hours awareness)
- Emergency response procedures
- Comprehensive statistics and health monitoring
- Export capabilities for analysis

**Optimization Modes**:
- **High Performance**: Maximum performance for market hours
- **Balanced**: Optimal balance of performance and efficiency
- **Power Save**: Energy-efficient operation for off-hours
- **Emergency**: Minimal resource usage during system stress

## Configuration

The system is configured through `cpu_config.yml`, which defines:

- **Core Allocation Policies**: Target cores for each priority level
- **QoS Mappings**: Workload types to QoS class assignments
- **Performance Targets**: Latency and throughput goals per operation
- **Thermal Management**: Temperature thresholds and response actions
- **Process Management**: Resource limits and scheduling policies
- **Monitoring Settings**: Metrics collection and alert thresholds

## API Integration

The system provides a comprehensive REST API (`optimization_routes.py`) for:

### System Health and Monitoring
- `GET /api/v1/optimization/health` - Current system health
- `GET /api/v1/optimization/stats` - Comprehensive statistics
- `GET /api/v1/optimization/core-utilization` - Per-core CPU usage
- `GET /api/v1/optimization/alerts` - Active performance alerts

### Process and Workload Management
- `POST /api/v1/optimization/register-process` - Register process for optimization
- `POST /api/v1/optimization/classify-workload` - Classify and optimize workload
- `POST /api/v1/optimization/rebalance-workloads` - Manual workload rebalancing

### Latency Measurement
- `POST /api/v1/optimization/start-latency-measurement` - Begin operation timing
- `POST /api/v1/optimization/end-latency-measurement/{id}` - Complete timing
- `GET /api/v1/optimization/latency-stats` - Latency statistics and percentiles

### System Control
- `GET /api/v1/optimization/optimization-mode` - Current optimization mode
- `POST /api/v1/optimization/optimization-mode` - Set optimization mode
- `POST /api/v1/optimization/export-performance-data` - Export metrics for analysis

## Performance Targets

The system is designed to achieve:

| Operation Type | Target Latency | Max Latency | Throughput |
|---|---|---|---|
| Order Execution | 0.5ms | 1.0ms | 10,000 ops/sec |
| Market Data Processing | 2.0ms | 5.0ms | 50,000 ops/sec |
| Risk Calculation | 5.0ms | 10.0ms | 1,000 ops/sec |
| Analytics Generation | 50ms | 100ms | 100 ops/sec |

## Usage Examples

### Basic Initialization
```python
from optimization import OptimizerController

# Initialize the optimization system
optimizer = OptimizerController()
optimizer.initialize()

# Register a trading process
optimizer.register_process(
    pid=12345,
    process_class=ProcessClass.TRADING_CORE,
    priority=WorkloadPriority.ULTRA_LOW_LATENCY
)
```

### Latency Measurement
```python
# Start measuring order execution latency
operation_id = optimizer.start_latency_measurement("order_execution")

# ... perform order execution ...

# End measurement
latency_ms = optimizer.end_latency_measurement(operation_id, success=True)
print(f"Order execution latency: {latency_ms:.2f}ms")
```

### Workload Classification
```python
# Classify a function for optimal scheduling
category, priority = optimizer.classify_and_optimize_workload(
    function_name="execute_market_order",
    module_name="trading_engine",
    execution_context={"latency_sensitive": True}
)
print(f"Classified as {category} with priority {priority}")
```

### GCD Task Dispatch
```python
# Dispatch a task using optimal QoS scheduling
task_id = optimizer.dispatch_task(
    queue_name="trading.orders",
    task_func=process_order,
    order_data=order
)
```

## Monitoring and Alerting

### Real-time Metrics
The system continuously monitors:
- Per-core CPU utilization with 100ms granularity
- Memory usage and pressure indicators
- Operation latency with microsecond precision
- Thermal state and power consumption
- Process resource consumption

### Alert Levels
- **Info**: General system information
- **Warning**: Performance degradation detected
- **Critical**: System stress, immediate attention required
- **Emergency**: System failure imminent, emergency procedures activated

### Emergency Procedures
When critical conditions are detected:
1. **CPU Overload**: Kill non-essential processes, migrate workloads
2. **Thermal Emergency**: Throttle CPU, suspend background tasks
3. **Memory Pressure**: Restart memory-intensive processes
4. **Deadlock Detection**: Kill deadlocked processes, restart services

## Production Deployment

### System Requirements
- macOS with Apple Silicon (M4 Max recommended)
- Python 3.9+
- Required packages: `psutil`, `numpy`, `scikit-learn`, `pyyaml`, `fastapi`
- Administrative privileges for process priority management

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure proper permissions for CPU affinity
sudo chown root:admin /usr/bin/taskpolicy
sudo chmod 755 /usr/bin/taskpolicy
```

### Integration with FastAPI
```python
from fastapi import FastAPI
from optimization.optimization_routes import router as optimization_router

app = FastAPI()
app.include_router(optimization_router)
```

### Monitoring Setup
```yaml
# docker-compose.yml addition
  optimization-monitor:
    build: 
      context: ./backend
      dockerfile: Dockerfile.optimization
    volumes:
      - ./backend/optimization:/app/optimization
    environment:
      - OPTIMIZATION_CONFIG_PATH=/app/optimization/cpu_config.yml
    ports:
      - "8080:8080"
```

## Testing and Validation

### Performance Benchmarks
The system includes comprehensive benchmarks to validate:
- Latency reduction (target: <1ms for order execution)
- Throughput improvement (target: 50x increase)
- Resource utilization optimization
- Thermal efficiency

### Load Testing
```python
# Run comprehensive load test
python -m optimization.tests.load_test --duration=300 --processes=100
```

### Validation Metrics
- Order execution latency percentiles (P50, P95, P99)
- CPU utilization distribution across cores
- Memory usage efficiency
- Alert response time and accuracy

## Advanced Features

### Machine Learning Integration
- Automatic workload pattern recognition
- Predictive CPU allocation based on market conditions
- Adaptive threshold tuning based on historical performance
- Anomaly detection for system health monitoring

### Market-Aware Optimization
- Market session-based mode switching
- High volatility performance boosting
- After-hours power efficiency optimization
- Holiday and weekend reduced operation modes

### Multi-Engine Coordination
Integration with Nautilus trading platform's 9 containerized engines:
- Analytics Engine: Optimized for batch processing
- Factor Engine: High-throughput data transformation
- ML Engine: GPU-accelerated model inference
- Risk Engine: Ultra-low latency risk calculations
- Portfolio Engine: Balance optimization and reporting

## Troubleshooting

### Common Issues

**High CPU Utilization**
```bash
# Check core distribution
curl http://localhost:8001/api/v1/optimization/core-utilization

# Manually rebalance
curl -X POST http://localhost:8001/api/v1/optimization/rebalance-workloads
```

**Latency Spikes**
```bash
# Check current alerts
curl http://localhost:8001/api/v1/optimization/alerts

# Get latency statistics
curl http://localhost:8001/api/v1/optimization/latency-stats?duration_minutes=5
```

**Memory Pressure**
```bash
# Check system health
curl http://localhost:8001/api/v1/optimization/health

# Switch to power save mode
curl -X POST http://localhost:8001/api/v1/optimization/optimization-mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "power_save"}'
```

### Debug Mode
```python
# Enable verbose logging
optimizer = OptimizerController()
optimizer.config["development"]["debug"]["enable_verbose_logging"] = True
```

### Performance Analysis
```python
# Export performance data for analysis
optimizer.export_performance_data(
    output_dir="/tmp/perf_analysis", 
    duration_hours=24
)
```

## Future Enhancements

- **Neural Accelerator Integration**: Leverage M4 Max neural engine for ML workloads
- **Metal Compute Optimization**: GPU compute scheduling for analytics
- **Power Management**: Advanced power state management
- **Cross-Machine Coordination**: Multi-node optimization for distributed trading
- **Real-time Visualization**: Web-based real-time performance dashboard

---

This CPU optimization system provides enterprise-grade performance management specifically tuned for ultra-low latency trading operations on M4 Max hardware, delivering measurable improvements in system responsiveness and resource efficiency.
# M4 Max Benchmarking Suite

Comprehensive benchmarking and validation suite for M4 Max optimizations in the Nautilus trading platform.

## Overview

This suite provides extensive performance testing and validation for all M4 Max optimizations including:

- **Metal GPU Acceleration** - 40 GPU cores with Metal Performance Shaders
- **Neural Engine Performance** - 16-core Neural Engine with Core ML
- **CPU Optimization** - 12 Performance cores + 4 Efficiency cores
- **Unified Memory Management** - 546 GB/s memory bandwidth
- **Container Optimization** - Docker optimization for ARM64 architecture
- **Trading Performance** - Ultra-low latency trading operations
- **AI/ML Inference** - Real-time ML model inference
- **System Stability** - Stress testing and thermal management

## Components

### 1. Performance Benchmark Suite (`performance_suite.py`)
Comprehensive benchmarks for all M4 Max optimizations:
- Metal GPU acceleration performance
- Neural Engine inference speed
- CPU core optimization validation
- Unified memory bandwidth testing
- System integration performance

### 2. Hardware Validation (`hardware_validation.py`)
M4 Max hardware detection and capability validation:
- M4 Max chip detection and verification
- Performance and Efficiency core validation
- GPU core availability and Metal framework testing
- Neural Engine availability and Core ML testing
- Memory architecture validation

### 3. Container Benchmarks (`container_benchmarks.py`)
Performance testing for containerized engines:
- Container startup time optimization
- Resource utilization efficiency
- Inter-container communication performance
- Docker optimization validation
- Container scaling performance

### 4. Trading Benchmarks (`trading_benchmarks.py`)
Trading-specific performance benchmarks:
- Order execution latency (<10ms target)
- Market data processing throughput (>50K messages/sec)
- Risk calculation performance (<5ms)
- Strategy execution benchmarks
- WebSocket streaming performance

### 5. AI/ML Benchmarks (`ai_benchmarks.py`)
Neural Engine and AI performance testing:
- Neural Engine inference speed and accuracy
- GPU-accelerated ML model training
- Core ML model conversion and deployment
- Trading model accuracy and performance
- Pattern recognition benchmarks

### 6. Stress Testing Suite (`stress_tests.py`)
System-wide stress testing and stability validation:
- System-wide stress testing under extreme loads
- Thermal management validation
- Emergency response testing
- Memory pressure testing
- High-frequency trading simulation

### 7. Benchmark Reporter (`benchmark_reporter.py`)
Comprehensive reporting with analytics:
- Performance reports with visualizations
- Before/after comparison analysis
- Performance regression detection
- Export to various formats (JSON, CSV, HTML)
- Historical trend analysis

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install numpy matplotlib seaborn pandas psutil docker

# For Neural Engine testing (optional)
pip install coremltools

# For Metal acceleration (requires PyTorch with MPS)
pip install torch torchvision torchaudio
```

### Basic Usage

```bash
# Run full benchmark suite
python run_benchmarks.py --suite all --format html

# Quick performance check
python run_benchmarks.py --suite performance --quick-run

# Run only trading benchmarks
python run_benchmarks.py --suite trading --output-dir ./results

# Hardware validation only
python run_benchmarks.py --suite hardware

# Regression testing
python run_benchmarks.py --regression-check
```

### Programmatic Usage

```python
import asyncio
from benchmarks.performance_suite import PerformanceBenchmarkSuite
from benchmarks.benchmark_reporter import BenchmarkReporter

async def run_performance_test():
    # Run performance benchmarks
    suite = PerformanceBenchmarkSuite()
    results = await suite.run_full_benchmark()
    
    # Generate report
    reporter = BenchmarkReporter()
    report = await reporter.generate_comprehensive_report(
        performance_results=results
    )
    
    print(f"Report generated: {report.report_id}")
    print(f"Overall performance score: {results.performance_improvement}")

# Run the benchmark
asyncio.run(run_performance_test())
```

## Command Line Options

```
--suite [all|performance|hardware|containers|trading|ai|stress]
    Benchmark suite to run (default: all)

--output-dir PATH
    Output directory for results (default: benchmark_results)

--format [json|html|csv]
    Output format(s) (default: html json)

--quick-run
    Run with reduced iterations for faster execution

--regression-check
    Run only regression tests against baselines

--export-baselines
    Save current results as new performance baselines

--verbose, -v
    Enable verbose logging

--log-file PATH
    Log file path (default: benchmark_run.log)
```

## Performance Targets (SLAs)

The benchmark suite validates against the following performance targets:

| Metric | Target | Description |
|--------|--------|-------------|
| Order Execution Latency | <10ms | Time to execute a single order |
| Market Data Throughput | >50K msgs/sec | Market data processing rate |
| Risk Calculation | <5ms | Portfolio risk calculation time |
| Strategy Execution | <20ms | Trading strategy execution time |
| WebSocket Latency | <2ms | WebSocket message processing |
| Neural Engine Inference | <5ms | Batch-32 inference time |
| Container Startup | <5sec | Container startup time |
| Memory Allocation | <10ms | 1GB memory allocation time |

## Benchmark Results

### Sample Performance Report

```
M4 Max Performance Benchmark Results
====================================
System: M4 Max (12P+4E cores, 40 GPU cores, 16 Neural Engine cores)
Memory: 128GB Unified Memory (546 GB/s bandwidth)
Platform: macOS 14.0 (ARM64)

Performance Summary:
- Average Latency: 3.2ms
- P95 Latency: 8.7ms
- P99 Latency: 15.3ms
- Average Throughput: 45,230 ops/sec
- Metal Acceleration: ENABLED
- Neural Engine: ACTIVE (87% utilization)

SLA Compliance:
✓ Order Execution: 2.1ms (target: <10ms)
✓ Market Data: 52,340 msgs/sec (target: >50K)
✓ Risk Calculation: 3.8ms (target: <5ms)
✓ Neural Inference: 4.2ms (target: <5ms)

Hardware Validation:
✓ M4 Max Detected: YES
✓ Metal Framework: AVAILABLE
✓ Neural Engine: 16 cores detected
✓ Unified Memory: 546 GB/s bandwidth confirmed

Optimization Status:
✓ CPU Affinity: Performance cores utilized
✓ Memory Management: Unified memory optimized
✓ GPU Acceleration: 40 cores active
✓ Container Optimization: ARM64 native

Recommendations:
- System performing within optimal ranges
- Consider increasing Neural Engine batch sizes
- Monitor thermal performance under sustained load
```

## Regression Testing

The suite automatically detects performance regressions by comparing results against historical baselines:

```bash
# Run regression check
python run_benchmarks.py --regression-check

# Update baselines after optimization
python run_benchmarks.py --suite performance --export-baselines
```

### Regression Thresholds

- **Performance Degradation**: >5% slower than baseline
- **Accuracy Drop**: >2% accuracy decrease
- **Throughput Reduction**: >10% throughput decrease
- **Latency Increase**: >20% latency increase
- **Error Rate Increase**: >1% error rate increase

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: M4 Max Performance Tests

on: [push, pull_request]

jobs:
  performance-tests:
    runs-on: [self-hosted, macOS, ARM64]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install torch torchvision torchaudio
    
    - name: Run performance benchmarks
      run: |
        python benchmarks/run_benchmarks.py --suite performance --quick-run
    
    - name: Run regression check
      run: |
        python benchmarks/run_benchmarks.py --regression-check
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results/
```

## Monitoring Integration

### Prometheus Metrics

The benchmark suite can export metrics to Prometheus:

```python
from benchmarks.performance_suite import PerformanceBenchmarkSuite
from prometheus_client import Gauge, push_to_gateway

# Run benchmarks
suite = PerformanceBenchmarkSuite()
results = await suite.run_full_benchmark()

# Export to Prometheus
latency_gauge = Gauge('trading_latency_ms', 'Trading operation latency')
latency_gauge.set(results.performance_summary['overall_avg_latency_ms'])

push_to_gateway('prometheus-gateway:9091', job='m4-max-benchmarks', registry=registry)
```

### Grafana Dashboard

Sample Grafana dashboard queries:

```promql
# Average latency trend
rate(trading_latency_ms[5m])

# Throughput monitoring
rate(trading_throughput_total[5m])

# Neural Engine utilization
neural_engine_utilization_percent

# System stability score
system_stability_score_percent
```

## Troubleshooting

### Common Issues

1. **Metal Not Available**
   ```
   Error: Metal acceleration not available
   Solution: Ensure running on Apple Silicon with macOS 12+
   ```

2. **Docker Not Running**
   ```
   Error: Docker client initialization failed
   Solution: Start Docker Desktop or Docker daemon
   ```

3. **Memory Allocation Failures**
   ```
   Error: Cannot allocate memory for stress test
   Solution: Close other applications or reduce test intensity
   ```

4. **Permission Errors**
   ```
   Error: Cannot access system information
   Solution: Run with appropriate permissions or disable affected tests
   ```

### Performance Optimization Tips

1. **For Best Performance**:
   - Ensure system is plugged in (not on battery)
   - Close unnecessary applications
   - Use latest macOS version
   - Enable performance mode if available

2. **For Consistent Results**:
   - Run multiple iterations
   - Use warmup periods
   - Monitor system temperature
   - Avoid running during system maintenance

## Contributing

1. **Adding New Benchmarks**:
   - Follow the existing pattern in benchmark modules
   - Include both optimized and fallback implementations
   - Add appropriate metadata and error handling
   - Update baselines if needed

2. **Extending Hardware Validation**:
   - Add new validation checks in `hardware_validation.py`
   - Include appropriate thresholds and recommendations
   - Test on various hardware configurations

3. **Improving Reports**:
   - Add new visualization types in `benchmark_reporter.py`
   - Extend HTML templates for better presentation
   - Add export formats as needed

## License

This benchmarking suite is part of the Nautilus trading platform and follows the same MIT license terms.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `benchmark_run.log`
3. Open an issue in the main Nautilus repository
4. Contact the development team

---

**Note**: This benchmarking suite is optimized for M4 Max hardware. Results on other systems may vary and some optimizations may not be available.
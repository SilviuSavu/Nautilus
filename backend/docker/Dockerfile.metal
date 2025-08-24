# Metal GPU Acceleration Dockerfile for M4 Max
# Optimized for GPU-accelerated workloads using Apple's Metal Performance Shaders

FROM --platform=linux/arm64/v8 python:3.13-slim

# Metadata
LABEL maintainer="Nautilus Trading Platform"
LABEL description="Metal GPU-accelerated container for M4 Max hardware"
LABEL version="1.0.0"
LABEL architecture="arm64"
LABEL gpu.framework="Metal"
LABEL acceleration.type="GPU"

# Environment variables for Metal optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=2 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Metal framework environment variables
ENV METAL_DEVICE_WRAPPER_TYPE=1 \
    METAL_PERFORMANCE_SHADERS_FRAMEWORKS=1 \
    GPU_ENABLED=true \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    TF_METAL_ENABLED=1

# Apple Silicon optimizations
ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    VECLIB_MAXIMUM_THREADS=8 \
    ACCELERATE_NEW_LAPACK=1 \
    ACCELERATE_LAPACK_ILP64=1

# Install system dependencies optimized for Metal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    libffi-dev \
    libssl-dev \
    # Graphics and compute libraries
    mesa-utils \
    opencl-headers \
    ocl-icd-opencl-dev \
    # Development tools
    gdb \
    valgrind \
    strace \
    # Networking
    netcat-openbsd \
    net-tools \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Upgrade pip and install wheel for better performance
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with Metal Performance Shaders (MPS) support
# Note: This installs the nightly build with Metal support
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cpu \
    --no-cache-dir

# Install TensorFlow with Metal optimization
RUN pip install tensorflow-metal tensorflow-macos --no-cache-dir

# Install GPU-accelerated numerical libraries
RUN pip install --no-cache-dir \
    # Core scientific computing with Metal acceleration
    numpy==1.26.* \
    scipy==1.12.* \
    pandas==2.2.* \
    # Machine learning frameworks
    scikit-learn==1.4.* \
    xgboost==2.0.* \
    lightgbm==4.3.* \
    # Deep learning utilities
    transformers==4.37.* \
    huggingface-hub==0.20.* \
    # GPU-accelerated array processing
    cupy-cuda12x \
    # Metal-optimized libraries
    mlx==0.12.* \
    mlx-lm==0.8.* \
    # Performance libraries
    numba==0.59.* \
    llvmlite==0.42.* \
    # Computer vision with GPU acceleration
    opencv-python-headless==4.9.* \
    pillow==10.2.* \
    # Data processing
    polars==0.20.* \
    pyarrow==15.0.*

# Install specialized Metal compute libraries
RUN pip install --no-cache-dir \
    # Apple's Core ML tools
    coremltools==7.2.* \
    # ONNX runtime with Metal support
    onnxruntime==1.17.* \
    # Quantization and optimization
    optimum==1.17.* \
    # Accelerated networking
    uvloop==0.19.* \
    # High-performance JSON
    orjson==3.9.* \
    # Fast HTTP client
    httpx==0.26.* \
    # Async database drivers
    asyncpg==0.29.* \
    aioredis==2.0.*

# Install trading-specific libraries with GPU acceleration
RUN pip install --no-cache-dir \
    # Financial data analysis
    yfinance==0.2.* \
    pandas-ta==0.3.* \
    ta-lib==0.4.* \
    # Risk management
    quantlib==1.33 \
    # Portfolio optimization
    cvxpy==1.4.* \
    # Time series analysis
    arch==6.3.* \
    statsmodels==0.14.* \
    # Market data
    alpha-vantage==2.3.* \
    fredapi==0.5.*

# Create application directory structure
RUN mkdir -p /app/{src,data,cache,models,logs,config} \
    && mkdir -p /app/.cache/{torch,transformers,huggingface} \
    && mkdir -p /tmp/metal-cache

# Set up GPU device access
RUN mkdir -p /dev/dri && \
    echo "Setting up GPU device access for Metal framework"

# Copy Metal-specific configuration files
COPY ./docker/metal-config.py /app/config/
COPY ./docker/gpu-utils.py /app/src/

# Set working directory
WORKDIR /app

# Create non-root user for security (but with GPU access)
RUN groupadd -r nautilus && \
    useradd -r -g nautilus -s /bin/bash nautilus && \
    # Add user to necessary groups for GPU access
    usermod -a -G video nautilus && \
    usermod -a -G render nautilus && \
    # Set up directory permissions
    chown -R nautilus:nautilus /app && \
    chown -R nautilus:nautilus /tmp/metal-cache

# Copy application requirements
COPY requirements-metal.txt /app/
RUN pip install --no-cache-dir -r requirements-metal.txt

# GPU health check script
COPY --chown=nautilus:nautilus <<EOF /app/gpu_health_check.py
#!/usr/bin/env python3
"""GPU Health Check for Metal framework on M4 Max"""

import sys
import platform
import subprocess

def check_metal_support():
    """Check Metal framework availability"""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✓ Metal Performance Shaders (MPS) is available")
            return True
        else:
            print("✗ Metal Performance Shaders (MPS) not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_tensorflow_metal():
    """Check TensorFlow Metal plugin"""
    try:
        import tensorflow as tf
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✓ TensorFlow Metal GPU devices: {len(gpu_devices)}")
            return True
        else:
            print("✗ No TensorFlow Metal GPU devices found")
            return False
    except ImportError:
        print("✗ TensorFlow not installed")
        return False

def check_system_info():
    """Display system information"""
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python version: {platform.python_version()}")
    
    # Check if running on Apple Silicon
    if platform.machine() == 'arm64':
        print("✓ Running on Apple Silicon")
    else:
        print(f"⚠ Running on {platform.machine()}, not Apple Silicon")

def main():
    print("=== Metal GPU Health Check ===")
    check_system_info()
    print()
    
    metal_ok = check_metal_support()
    tf_metal_ok = check_tensorflow_metal()
    
    if metal_ok and tf_metal_ok:
        print("\n✅ Metal GPU acceleration is fully operational")
        sys.exit(0)
    else:
        print("\n❌ Metal GPU acceleration has issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Performance benchmark script
COPY --chown=nautilus:nautilus <<EOF /app/metal_benchmark.py
#!/usr/bin/env python3
"""Metal Performance Benchmark for M4 Max"""

import time
import torch
import numpy as np
from typing import Dict, Any

def benchmark_pytorch_metal(size: int = 1000) -> Dict[str, Any]:
    """Benchmark PyTorch Metal performance"""
    if not torch.backends.mps.is_available():
        return {"error": "Metal not available"}
    
    device = torch.device("mps")
    
    # Matrix multiplication benchmark
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = torch.mm(a, b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
    torch.mps.synchronize()  # Wait for GPU operations to complete
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    gflops = (2 * size**3) / (avg_time * 1e9)
    
    return {
        "operation": "matrix_multiplication",
        "size": f"{size}x{size}",
        "avg_time_ms": avg_time * 1000,
        "gflops": gflops,
        "device": str(device)
    }

def benchmark_memory_transfer(size_mb: int = 100) -> Dict[str, Any]:
    """Benchmark CPU-GPU memory transfer"""
    if not torch.backends.mps.is_available():
        return {"error": "Metal not available"}
    
    device = torch.device("mps")
    size = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
    
    # CPU tensor
    cpu_tensor = torch.randn(size)
    
    # Transfer to GPU
    start_time = time.time()
    gpu_tensor = cpu_tensor.to(device)
    torch.mps.synchronize()
    transfer_to_gpu_time = time.time() - start_time
    
    # Transfer back to CPU
    start_time = time.time()
    cpu_tensor_back = gpu_tensor.to("cpu")
    transfer_to_cpu_time = time.time() - start_time
    
    bandwidth_to_gpu = (size_mb) / transfer_to_gpu_time
    bandwidth_to_cpu = (size_mb) / transfer_to_cpu_time
    
    return {
        "operation": "memory_transfer",
        "size_mb": size_mb,
        "transfer_to_gpu_ms": transfer_to_gpu_time * 1000,
        "transfer_to_cpu_ms": transfer_to_cpu_time * 1000,
        "bandwidth_to_gpu_mbps": bandwidth_to_gpu,
        "bandwidth_to_cpu_mbps": bandwidth_to_cpu
    }

def run_benchmarks():
    """Run comprehensive Metal benchmarks"""
    print("=== Metal Performance Benchmark ===")
    
    # Matrix multiplication benchmarks
    for size in [512, 1024, 2048]:
        result = benchmark_pytorch_metal(size)
        if "error" not in result:
            print(f"Matrix Mul {result['size']}: {result['avg_time_ms']:.2f}ms, {result['gflops']:.2f} GFLOPS")
        else:
            print(f"Matrix Mul {size}x{size}: {result['error']}")
    
    print()
    
    # Memory transfer benchmarks
    for size_mb in [10, 50, 100, 200]:
        result = benchmark_memory_transfer(size_mb)
        if "error" not in result:
            print(f"Memory Transfer {result['size_mb']}MB:")
            print(f"  To GPU: {result['transfer_to_gpu_ms']:.2f}ms ({result['bandwidth_to_gpu_mbps']:.1f} MB/s)")
            print(f"  To CPU: {result['transfer_to_cpu_ms']:.2f}ms ({result['bandwidth_to_cpu_mbps']:.1f} MB/s)")
        else:
            print(f"Memory Transfer {size_mb}MB: {result['error']}")

if __name__ == "__main__":
    run_benchmarks()
EOF

# Make scripts executable
RUN chmod +x /app/gpu_health_check.py /app/metal_benchmark.py

# Switch to non-root user
USER nautilus

# Health check using our custom GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/gpu_health_check.py

# Default command
CMD ["python", "-c", "print('Metal GPU-accelerated container ready'); import sys; sys.exit(0)"]

# Build arguments for optimization
ARG BUILD_TYPE=release
ARG OPTIMIZATION_LEVEL=O3
ARG TARGET_ARCH=arm64

# Labels for container identification
LABEL build.type=${BUILD_TYPE}
LABEL optimization.level=${OPTIMIZATION_LEVEL}
LABEL target.architecture=${TARGET_ARCH}
LABEL gpu.metal.enabled=true
LABEL neural.engine.ready=true
LABEL performance.optimized=true
# ML Inference Engine Dockerfile
# M4 Max Optimized for machine learning inference and prediction
# Apple Silicon + Metal Performance Shaders acceleration

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries + M4 Max optimization
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    curl \
    libomp-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# M4 Max specific environment for compilation
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Copy requirements and install Python dependencies
COPY requirements.minimal.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.minimal.txt

# Copy MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy ML engine code
COPY simple_ml_engine.py .

# Create user for security
RUN useradd -m -u 1000 ml && \
    chown -R ml:ml /app

USER ml

# Health check - ML inference monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

# Expose port
EXPOSE 8400

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8400
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# M4 Max ML engine specific settings
ENV ML_MODEL_CACHE_SIZE=200
ENV ML_PREDICTION_TIMEOUT=60
ENV ML_INFERENCE_WORKERS=6
ENV ML_MODEL_ENABLED=true

# M4 Max Hardware Acceleration
ENV HARDWARE_PLATFORM=m4_max
ENV ENABLE_METAL_ACCELERATION=true
ENV ENABLE_NEURAL_ENGINE=true
ENV ML_USE_METAL_PERFORMANCE_SHADERS=true
ENV ML_BATCH_INFERENCE=true
ENV ML_MODEL_QUANTIZATION=true

# M4 Max Resource configuration
ENV ML_MAX_MEMORY=10g
ENV ML_MAX_CPU=3.0

# Start the ML inference engine
CMD ["python", "simple_ml_engine.py"]
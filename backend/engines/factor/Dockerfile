# Factor Engine Dockerfile
# M4 Max Optimized for high-volume factor calculations
# Maximized parallel processing for Apple Silicon

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing + M4 Max optimization
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libomp-dev \
    pkg-config \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# M4 Max heavy computation optimization
ENV OPENBLAS_NUM_THREADS=6
ENV OMP_NUM_THREADS=6
ENV NUMEXPR_NUM_THREADS=6

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client and clock
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy factor engine code
COPY factor_engine.py .

# Create user for security
RUN useradd -m -u 1000 factor && \
    chown -R factor:factor /app

USER factor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8300/health || exit 1

# Expose port
EXPOSE 8300

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8300
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# M4 Max Factor engine specific settings
ENV FACTOR_CALCULATION_WORKERS=6
ENV FACTOR_BATCH_SIZE=100
ENV CACHE_CLEANUP_INTERVAL=300

# M4 Max Hardware Acceleration
ENV HARDWARE_PLATFORM=m4_max
ENV ENABLE_METAL_ACCELERATION=true
ENV ENABLE_NEURAL_ENGINE=true
ENV FACTOR_PARALLEL_PROCESSING=true

# M4 Max Resource configuration
ENV FACTOR_MAX_MEMORY=12g
ENV FACTOR_MAX_CPU=4.0

# Start the factor engine
CMD ["python", "factor_engine.py"]

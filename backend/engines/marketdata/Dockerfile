# Market Data Engine Dockerfile
# M4 Max Optimized for ultra-low latency market data processing
# Apple Silicon + CPU optimization for real-time data feeds

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for data processing + M4 Max optimization
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

# M4 Max optimization environment for ultra-low latency
ENV OPENBLAS_NUM_THREADS=2
ENV OMP_NUM_THREADS=2

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy market data engine code
COPY simple_marketdata_engine.py .

# Create user for security
RUN useradd -m -u 1000 marketdata && \
    chown -R marketdata:marketdata /app

USER marketdata

# Health check - market data feed monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8800/health || exit 1

# Expose port
EXPOSE 8800

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8800
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Market data engine specific settings
ENV MARKET_DATA_CACHE_SIZE=10000
ENV DATA_RETENTION_HOURS=24
ENV MAX_SYMBOLS_TRACKED=1000
ENV FEED_UPDATE_INTERVAL_MS=100
ENV LATENCY_ALERT_THRESHOLD_MS=50

# M4 Max Hardware Acceleration for ultra-low latency
ENV HARDWARE_PLATFORM=m4_max
ENV ENABLE_CPU_OPTIMIZATION=true
ENV MARKETDATA_WORKERS=2
ENV LOW_LATENCY_MODE=true
ENV FEED_PROCESSING_BATCH_SIZE=100

# M4 Max Resource configuration
ENV MARKETDATA_MAX_MEMORY=3g
ENV MARKETDATA_MAX_CPU=2.0

# Start the market data engine
CMD ["python", "simple_marketdata_engine.py"]
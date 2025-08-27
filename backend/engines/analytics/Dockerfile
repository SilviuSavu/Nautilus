# Analytics Engine Dockerfile
# M4 Max Optimized for high-performance analytics processing
# Apple Silicon + Metal acceleration for computational analytics

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for analytics + M4 Max optimization
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

# M4 Max optimization environment
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client, MarketData client, and clock system
COPY enhanced_messagebus_client.py .
COPY ../../marketdata_client.py .
COPY ../../universal_enhanced_messagebus_client.py .
COPY clock.py .

# Copy analytics engine code and real data integration
COPY simple_analytics_engine.py .
COPY real_data_integration.py .

# Create user for security
RUN useradd -m -u 1000 analytics && \
    chown -R analytics:analytics /app

USER analytics

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Expose port
EXPOSE 8100

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8100
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# MarketData Hub configuration (MANDATORY)
ENV MARKETDATA_HUB_ENABLED=true
ENV DIRECT_API_CALLS_BLOCKED=true
ENV MARKETDATA_ENGINE_URL=http://marketdata-engine:8800

# M4 Max Hardware Acceleration
ENV HARDWARE_PLATFORM=m4_max
ENV ENABLE_METAL_ACCELERATION=true
ENV ENABLE_NEURAL_ENGINE=true
ENV ANALYTICS_WORKERS=4
ENV BATCH_PROCESSING_SIZE=1000

# M4 Max Resource configuration
ENV ANALYTICS_MAX_MEMORY=6g
ENV ANALYTICS_MAX_CPU=3.0

# Start the analytics engine
CMD ["python", "simple_analytics_engine.py"]
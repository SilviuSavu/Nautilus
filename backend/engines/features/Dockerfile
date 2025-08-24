# Feature Engineering Engine Dockerfile
# Optimized for high-performance feature calculation

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for numerical computation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy features engine code
COPY simple_features_engine.py .

# Create user for security
RUN useradd -m -u 1000 features && \
    chown -R features:features /app

USER features

# Health check - feature calculation monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8500/health || exit 1

# Expose port
EXPOSE 8500

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8500
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Feature engine specific settings
ENV FEATURE_CACHE_SIZE=1000
ENV FEATURE_CALC_TIMEOUT=30
ENV FEATURE_WORKERS=8
ENV TECHNICAL_FEATURES_ENABLED=true
ENV FUNDAMENTAL_FEATURES_ENABLED=true

# Resource limits (will be overridden by docker-compose)
ENV FEATURES_MAX_MEMORY=4g
ENV FEATURES_MAX_CPU=3.0

# Start the feature engineering engine
CMD ["python", "simple_features_engine.py"]
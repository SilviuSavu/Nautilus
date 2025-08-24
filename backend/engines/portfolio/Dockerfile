# Institutional Portfolio Engine Dockerfile
# M4 Max Optimized for institutional-grade portfolio management
# Enhanced with ArcticDB, VectorBT, and Risk Engine integration

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for institutional libraries + M4 Max optimization
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
    build-essential \
    cmake \
    git \
    libhdf5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# M4 Max optimization environment
ENV OPENBLAS_NUM_THREADS=4
ENV OMP_NUM_THREADS=4

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy enhanced MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy market data client for real data integration
COPY market_data_client.py .

# Copy all institutional portfolio engine components
COPY enhanced_portfolio_engine.py .
COPY risk_engine_integration.py .
COPY multi_portfolio_manager.py .
COPY portfolio_dashboard.py .
COPY institutional_portfolio_engine.py .

# Copy original simple engine for backward compatibility
COPY simple_portfolio_engine.py .

# Create user for security
RUN useradd -m -u 1000 portfolio && \
    chown -R portfolio:portfolio /app

USER portfolio

# Health check - portfolio optimization monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8900/health || exit 1

# Expose port
EXPOSE 8900

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8900
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Institutional Portfolio Engine Settings
ENV OPTIMIZATION_TIMEOUT=300
ENV MAX_PORTFOLIOS=1000
ENV REBALANCE_CHECK_INTERVAL=3600
ENV RISK_FREE_RATE=0.02
ENV OPTIMIZATION_CACHE_TTL=1800

# Enhanced Institutional Features
ENV ARCTICDB_URI=lmdb://./arctic_data
ENV VECTORBT_GPU_ENABLED=false
ENV RISK_ENGINE_URL=http://risk-engine:8200
ENV ENABLE_RISK_MONITORING=true
ENV FAMILY_OFFICE_SUPPORT=true
ENV INSTITUTIONAL_TIER=true

# Multi-Portfolio Management
ENV MAX_CLIENTS=100
ENV MAX_STRATEGIES_PER_PORTFOLIO=10
ENV MAX_PORTFOLIOS_PER_CLIENT=50
ENV GOAL_TRACKING_ENABLED=true
ENV TAX_OPTIMIZATION_ENABLED=true

# M4 Max Hardware Acceleration
ENV HARDWARE_PLATFORM=m4_max
ENV ENABLE_METAL_ACCELERATION=true
ENV ENABLE_NEURAL_ENGINE=true
ENV PORTFOLIO_OPTIMIZATION_WORKERS=4
ENV OPTIMIZATION_BATCH_SIZE=500

# M4 Max Resource Configuration
ENV PORTFOLIO_MAX_MEMORY=8g
ENV PORTFOLIO_MAX_CPU=4.0

# Dashboard & Visualization
ENV ENABLE_DASHBOARD=true
ENV DASHBOARD_REFRESH_INTERVAL=300
ENV EXPORT_FORMATS=json,html

# Engine Selection (institutional by default, fallback to simple)
ENV ENGINE_MODE=institutional

# Start the institutional portfolio engine
CMD ["python", "institutional_portfolio_engine.py"]
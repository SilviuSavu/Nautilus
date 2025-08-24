# Risk Engine Dockerfile
# Optimized for critical latency risk management

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for risk calculations
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
COPY requirements.minimal.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.minimal.txt

# Copy MessageBus client and clock
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy modular engine components (NEW)
COPY risk_engine.py .
COPY engine.py .
COPY models.py .
COPY services.py .
COPY routes.py .

# Copy all Epic components
COPY advanced_risk_analytics.py .
COPY hybrid_risk_analytics.py .
COPY pyfolio_integration.py .
COPY portfolio_optimizer_client.py .
COPY supervised_knn_optimizer.py .
COPY professional_risk_reporter.py .
COPY distance_metrics.py .
COPY market_features.py .
COPY automated_report_scheduler.py .

# Copy enhanced risk engine components (NEW - August 2025)
COPY vectorbt_integration.py .
COPY arcticdb_client.py .
COPY ore_gateway.py .
COPY qlib_integration.py .
COPY hybrid_risk_processor.py .
COPY enterprise_risk_dashboard.py .
COPY enhanced_risk_api.py .

# Copy report templates
COPY report_templates/ ./report_templates/

# Copy simple engine for compatibility
COPY simple_risk_engine.py .

# Create directories for enhanced components (NEW - August 2025)
RUN mkdir -p /app/data/arctic && \
    mkdir -p /app/configs/ore && \
    mkdir -p /app/models/qlib && \
    mkdir -p /app/logs/enhanced && \
    mkdir -p /app/temp/backtests

# Create user for security
RUN useradd -m -u 1000 risk && \
    chown -R risk:risk /app

USER risk

# Health check - critical for risk management
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
    CMD curl -f http://localhost:8200/health || exit 1

# Expose port
EXPOSE 8200

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8200
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Risk engine specific settings
ENV RISK_CHECK_INTERVAL=5
ENV BREACH_PREDICTION_THRESHOLD=0.7
ENV ML_MODEL_ENABLED=true

# Portfolio Optimizer API configuration (Story 2.1)
ENV PORTFOLIO_OPTIMIZER_API_KEY=EgyPyGQSVQV4GMKWp5l7tf21wQOUaaw
ENV PORTFOLIO_OPTIMIZER_BASE_URL=https://api.portfoliooptimizer.io/v1
ENV PORTFOLIO_OPTIMIZER_TIMEOUT=30
ENV PORTFOLIO_OPTIMIZER_CACHE_TTL=300
ENV PORTFOLIO_OPTIMIZER_MAX_RETRIES=3
ENV PORTFOLIO_OPTIMIZER_CIRCUIT_BREAKER_THRESHOLD=5

# Enhanced risk engine configuration (NEW - August 2025)
ENV VECTORBT_GPU_ENABLED=true
ENV VECTORBT_CACHE_SIZE=1000
ENV ARCTICDB_STORAGE_PATH=/app/data/arctic
ENV ARCTICDB_MAX_CONNECTIONS=20
ENV ORE_GATEWAY_CONFIG_PATH=/app/configs/ore
ENV ORE_CACHE_TTL=1800
ENV QLIB_MODEL_PATH=/app/models/qlib
ENV QLIB_USE_NEURAL_ENGINE=true
ENV HYBRID_PROCESSOR_WORKERS=8
ENV HYBRID_ROUTING_ENABLED=true
ENV DASHBOARD_CACHE_TTL=300
ENV DASHBOARD_MAX_SYMBOLS=100

# M4 Max hardware acceleration for risk engine (NEW)
ENV M4_MAX_OPTIMIZED=1
ENV NEURAL_ENGINE_ENABLED=1
ENV METAL_GPU_ENABLED=1
ENV AUTO_HARDWARE_ROUTING=1
ENV HYBRID_ACCELERATION=1
ENV NEURAL_ENGINE_PRIORITY=HIGH
ENV RISK_USE_GPU_ACCELERATION=true

# Resource limits (enhanced for new components)
ENV RISK_MAX_MEMORY=2g
ENV RISK_MAX_CPU=1.0

# Start the complete risk engine with all Epic components
CMD ["python", "risk_engine.py"]

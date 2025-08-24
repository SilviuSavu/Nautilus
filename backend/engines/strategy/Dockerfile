# Strategy Engine Dockerfile
# Optimized for automated strategy deployment and testing

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for strategy deployment
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy strategy engine code
COPY simple_strategy_engine.py .

# Create user for security
RUN useradd -m -u 1000 strategy && \
    chown -R strategy:strategy /app

USER strategy

# Health check - strategy deployment monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8700/health || exit 1

# Expose port
EXPOSE 8700

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8700
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Strategy engine specific settings
ENV STRATEGY_TIMEOUT=300
ENV MAX_CONCURRENT_DEPLOYMENTS=5
ENV PIPELINE_STAGES="syntax_check,unit_tests,backtest,paper_trading,risk_validation"
ENV AUTO_ROLLBACK_ENABLED=true
ENV DEPLOYMENT_APPROVAL_REQUIRED=false

# Resource limits (will be overridden by docker-compose)
ENV STRATEGY_MAX_MEMORY=2g
ENV STRATEGY_MAX_CPU=1.0

# Start the strategy deployment engine
CMD ["python", "simple_strategy_engine.py"]
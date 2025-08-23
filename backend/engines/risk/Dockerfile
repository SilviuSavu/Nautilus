# Risk Management Engine Container
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install enhanced MessageBus client and risk dependencies
RUN pip install --no-cache-dir \
    redis[hiredis] \
    asyncio \
    numpy \
    pandas \
    scikit-learn \
    pydantic

# Copy engine code
COPY risk_engine.py .
COPY risk_management/ ./risk_management/
COPY risk/ ./risk/
COPY messagebus_client.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_NAME=risk
ENV MESSAGEBUS_TOPICS="risk.*,trading.orders,trading.positions"
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV RISK_CHECK_INTERVAL=5
ENV MAX_PORTFOLIO_EXPOSURE=1000000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the risk management engine
CMD ["python", "risk_engine.py"]
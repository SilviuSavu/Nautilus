# Analytics Engine Container
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

# Install enhanced MessageBus client
RUN pip install --no-cache-dir \
    redis[hiredis] \
    asyncio \
    numpy \
    pandas \
    pydantic

# Copy engine code
COPY analytics_engine.py .
COPY analytics/ ./analytics/
COPY messagebus_client.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_NAME=analytics
ENV MESSAGEBUS_TOPICS="analytics.*,risk.alerts,trading.executions"
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the analytics engine
CMD ["python", "analytics_engine.py"]
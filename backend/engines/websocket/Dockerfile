# WebSocket Engine Dockerfile
# Optimized for high-concurrency real-time streaming

FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for WebSocket handling
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy MessageBus client and clock system
COPY enhanced_messagebus_client.py .
COPY clock.py .

# Copy WebSocket engine code
COPY simple_websocket_engine.py .

# Create user for security
RUN useradd -m -u 1000 websocket && \
    chown -R websocket:websocket /app

USER websocket

# Health check - WebSocket service monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8600/health || exit 1

# Expose port
EXPOSE 8600

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
ENV PORT=8600
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# WebSocket engine specific settings
ENV WS_MAX_CONNECTIONS=1000
ENV WS_HEARTBEAT_INTERVAL=30
ENV WS_CLEANUP_INTERVAL=60
ENV WS_CONNECTION_TIMEOUT=300
ENV WS_MESSAGE_QUEUE_SIZE=10000

# Resource limits (will be overridden by docker-compose)
ENV WEBSOCKET_MAX_MEMORY=2g
ENV WEBSOCKET_MAX_CPU=1.0

# Start the WebSocket streaming engine
CMD ["python", "simple_websocket_engine.py"]
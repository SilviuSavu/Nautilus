# ML Inference Engine Container
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install enhanced MessageBus client and ML dependencies
RUN pip install --no-cache-dir \
    redis[hiredis] \
    asyncio \
    numpy \
    pandas \
    scikit-learn \
    torch \
    scipy \
    joblib \
    pydantic

# Copy engine code
COPY ml_inference_engine.py .
COPY ml/ ./ml/
COPY messagebus_client.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_NAME=ml_inference
ENV MESSAGEBUS_TOPICS="ml.*,market.regime,factor.scores"
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV ML_MODEL_PATH=/app/models
ENV INFERENCE_BATCH_SIZE=100

# Create models directory
RUN mkdir -p /app/models

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the ML inference engine
CMD ["python", "ml_inference_engine.py"]
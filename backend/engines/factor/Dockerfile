# Factor Synthesis Engine Container
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies including TA-Lib dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/ta-lib*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install enhanced MessageBus client and factor dependencies
RUN pip install --no-cache-dir \
    redis[hiredis] \
    asyncio \
    numpy \
    pandas \
    TA-Lib \
    scipy \
    pydantic

# Copy engine code
COPY factor_engine.py .
COPY factor_engine_service.py .
COPY cross_source_factor_engine.py .
COPY messagebus_client.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENGINE_NAME=factor
ENV MESSAGEBUS_TOPICS="factor.*,market.data.*,economic.data.*"
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV FACTOR_CALCULATION_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run the factor synthesis engine
CMD ["python", "factor_engine.py"]
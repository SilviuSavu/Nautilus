# Nautilus Volatility Forecasting Engine
# M4 Max optimized container for volatility modeling

FROM python:3.11-slim

# Set M4 Max optimization flags
ENV M4_MAX_OPTIMIZED=1 \
    METAL_ACCELERATION=1 \
    NEURAL_ENGINE_ENABLED=1 \
    CPU_OPTIMIZATION=1 \
    AUTO_HARDWARE_ROUTING=1

# Set working directory
WORKDIR /app

# Install system dependencies for M4 Max acceleration
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with M4 Max optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy volatility engine source code
COPY ../../../volatility/ ./volatility/
COPY main.py .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Set permissions
RUN chmod +x main.py

# Health check script
COPY healthcheck.py .
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Expose the volatility engine port
EXPOSE 9000

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    VOLATILITY_PORT=9000 \
    LOG_LEVEL=INFO \
    WORKERS=4

# Run the volatility engine
CMD ["python", "main.py"]
# Collateral Management Engine Dockerfile
# Optimized for M4 Max with hardware acceleration support

FROM python:3.13-slim

# M4 Max optimization flags
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV M4_MAX_OPTIMIZED=1
ENV METAL_ACCELERATION=1
ENV NEURAL_ENGINE_ENABLED=1
ENV CPU_OPTIMIZATION=1

# Install system dependencies for M4 Max optimization
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy collateral engine source
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash collateral && \
    chown -R collateral:collateral /app
USER collateral

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/collateral/health || exit 1

# Expose port
EXPOSE 8000

# Start the collateral management engine
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
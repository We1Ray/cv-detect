# Multi-stage build for CV Defect Detection API Server
FROM python:3.11-slim AS base

# Security: don't run as root
RUN groupadd -r cvdetect && useradd -r -g cvdetect -m cvdetect

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy application code
COPY dl_anomaly/ dl_anomaly/
COPY variation_model/ variation_model/
COPY shared/ shared/
COPY api/ api/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p results models && chown -R cvdetect:cvdetect /app

USER cvdetect

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Resource limits are set in docker-compose
CMD ["python", "-m", "api.server", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage Dockerfile for Financial Transaction Risk & Operations Intelligence Platform
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install system dependencies (including libgomp1 for LightGBM/XGBoost, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, precomputed summaries, reports, and documentation
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/
COPY reports/ reports/
COPY docs/ docs/
COPY sql/ sql/

# Expose ports: 8000 for FastAPI Real-Time Risk Engine, 8501 for Streamlit BI Dashboard
EXPOSE 8000 8501

# Default health check on FastAPI service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default command: launch FastAPI Risk Engine microservice
CMD ["uvicorn", "financial_risk.service:app", "--host", "0.0.0.0", "--port", "8000"]

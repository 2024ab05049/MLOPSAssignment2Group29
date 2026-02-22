# Multi-stage Dockerfile for Cats vs Dogs Classifier API
# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM base as production

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best_model.pth
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage build for Baby Cry Classifier
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all project files needed for build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies (builds the local package as well)
RUN uv sync --frozen --no-dev

# Copy config and models (only SVM artifacts, not large checkpoints)
COPY config.yml ./
RUN mkdir -p models
COPY models/svm_baby_cry.pkl models/scaler.pkl models/label_names.pkl ./models/

# Expose port
EXPOSE 8000

# Default environment variables
ENV SERVICE_MODE=monolith
ENV CONFIG_PATH=config.yml

# Run the application
CMD ["uv", "run", "uvicorn", "baby_cry_classifier.serve:app", "--host", "0.0.0.0", "--port", "8000"]

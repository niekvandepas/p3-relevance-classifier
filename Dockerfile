FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Copy uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*

# Use /app so RunPod's /workspace mount doesn't overwrite code
WORKDIR /app

# Copy dependency definitions first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen

# Copy the rest of your project files
COPY . .

# Automatically activate the uv virtual environment so 'python' commands use it
ENV PATH="/workspace/p3-relevance-classifier/.venv/bin:$PATH"

# Default command (RunPod overrides this for Jupyter/SSH, but good as fallback)
CMD ["sleep", "infinity"]

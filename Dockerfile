FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Copy uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Vim for editing files in a pinch, and zstd as it's required by Ollama's installer
RUN apt-get update && \
    apt-get install -y vim zstd && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# Use /app so RunPod's /workspace mount doesn't overwrite code
WORKDIR /app

# Copy dependency definitions first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen

# Default values for .env
RUN printf "LLM_NAME=bramvanroy/geitje-7b-ultra:q8_0\nLLM_REDDIT_DATA_SOURCE=huggingface\nHF_TOKEN=<token>\n" > .env

# Copy the rest of your project files
COPY . .

# Default command (RunPod overrides this for Jupyter/SSH, but good as fallback)
CMD ["sleep", "infinity"]

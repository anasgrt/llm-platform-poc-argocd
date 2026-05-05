# Use Python slim image (same base as other services) with pre-built llama-cpp-python wheels
# This avoids ARM64 compilation issues
FROM python:3.11-slim-bookworm

# Configure pip with longer timeouts and retries for unstable networks
RUN pip config set global.timeout 120 && \
    pip config set global.retries 5

# Install runtime libraries for OpenBLAS/OpenMP (wheels are pre-built, but runtime libs are needed)
# Use cached package lists if network is unavailable during build
RUN (apt-get update || apt-get update || true) && \
    (apt-get install -y --no-install-recommends \
    libopenblas-dev libomp-dev libgomp1 \
    || echo "Warning: Package installation failed, continuing with existing packages") \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install llama-cpp-python
# Using a version with better ARM64 wheel availability
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install with retries for network resilience
RUN for i in {1..5}; do \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "llama-cpp-python[server]" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    && break; \
    echo "pip install attempt $i failed, retrying in 10 seconds..."; \
    sleep 10; \
  done

# Model weights mounted at /models via PVC
VOLUME /models
EXPOSE 8080
WORKDIR /models

# Serve Qwen3 4B with OpenAI-compatible API
ENTRYPOINT ["python", "-m", "llama_cpp.server", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--n_ctx", "8192", \
    "--n_threads", "4"]

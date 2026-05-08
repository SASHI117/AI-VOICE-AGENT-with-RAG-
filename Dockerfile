# =====================================================
# Pipecat Voice Agent — Azure Container Apps image
# =====================================================

FROM python:3.12-slim

WORKDIR /app

# System dependencies:
#   ffmpeg → required by Pipecat audio resampler / ConvTasNet
#   curl   → used by container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Install dependencies using uv lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Ensure the virtual environment is used
ENV PATH="/app/.venv/bin:$PATH"

# Application code
COPY apsagent.py .
COPY apsagent_system_prompt.txt .
COPY convtasnet_processor.py .
COPY knowledge_base/ ./knowledge_base/
COPY rag_sharekit/ ./rag_sharekit/
COPY rag_version_2/ ./rag_version_2/
COPY smartflo_sharekit/ ./smartflo_sharekit/
# Also copy cache so we don't rebuild embeddings on container start
COPY .rag_cache/ ./.rag_cache/

# Server config
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check — Container Apps uses this to know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the Pipecat runner server
CMD ["python", "apsagent.py", "--host", "0.0.0.0", "--port", "8000"]
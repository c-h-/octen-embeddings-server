# octen-embeddings-server Dockerfile
#
# IMPORTANT: MLX requires Apple Silicon (M1/M2/M3/M4) with Metal support.
# Docker containers do not have Metal GPU access, so this image runs in
# CPU-fallback mode with significantly reduced performance.
#
# For production use on Apple Silicon, native installation is strongly
# recommended over Docker. See README.md for native setup instructions.
#
# This Dockerfile is provided for:
#   - CI/CD pipelines (linting, testing without model)
#   - Development environments
#   - Reference for deployment packaging

FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY server.py .
COPY convert_model.py .

# Model weights must be mounted at runtime:
#   docker run -v /path/to/models:/app/models ...
VOLUME /app/models

ENV HOST=0.0.0.0
ENV PORT=8100
ENV MLX_MODEL_PATH=/app/models/Octen-Embedding-8B-mlx

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

CMD ["python", "server.py"]

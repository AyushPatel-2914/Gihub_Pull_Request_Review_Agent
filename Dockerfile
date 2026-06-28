# ─────────────────────────────────────────────────────────────
# PR Review Agent — Webhook Server Container (Phase 5B)
#
# Build:  docker build -t pr-review-agent .
# Run:    docker run -p 8000:8000 --env-file .env pr-review-agent
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Avoid interactive prompts and reduce image size
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models at build time so
# the first webhook isn't slow. Comment out to download lazily.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)" || true
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

# Copy the application code
COPY src/ ./src/

# The server listens on 8000
EXPOSE 8000

# Run the webhook server
# PORT env var support for platforms like Render/Railway
CMD uvicorn server.webhook_server:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir src

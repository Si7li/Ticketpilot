# syntax=docker/dockerfile:1

# ---------- Stage 1: build the virtualenv and bake the embedding model ----------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/app/.venv FASTEMBED_CACHE_DIR=/opt/fastembed
WORKDIR /app

# Dependencies first (cached layer), then the project itself (editable, at /app/src).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras

# Pre-download the fastembed model so containers start offline.
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_DIR=/opt/fastembed HF_HUB_OFFLINE=1 \
    BUSINESS_DB_PATH=/app/var/ticketpilot.db CHECKPOINT_DB_PATH=/app/var/checkpoints.db
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/fastembed /opt/fastembed
COPY src ./src
COPY data/kb ./data/kb
COPY data/seed ./data/seed
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN mkdir -p /app/var && chown -R app:app /app/var /opt/fastembed && chmod +x /usr/local/bin/entrypoint.sh
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "ticketpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

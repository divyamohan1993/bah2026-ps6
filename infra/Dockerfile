# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# AgriStress serving image — Google Cloud Run-ready.
#
# Builds a slim FastAPI/Uvicorn image that serves the O(1) read-hot-path API
# (agristress.serving.api:app) plus the static MapLibre dashboard. It installs
# ONLY the pure-wheel `[serving]` extra (fastapi/uvicorn/httpx/typer/pillow) so
# the image stays small and builds without system GDAL/PROJ or torch.
#
# Cloud Run contract this image satisfies:
#   * listens on 0.0.0.0:$PORT  (Cloud Run injects $PORT, default 8080)
#   * stateless                  (state -> Memorystore via REDIS_URL; gold -> GCS)
#   * non-root, single process   (Cloud Run scales by *instances*, so 1 worker)
#
# Build (from repo root, build context = repo root):
#   docker build -f infra/Dockerfile -t agristress-api .
# Run locally:
#   docker run --rm -p 8080:8080 agristress-api
#   curl localhost:8080/health
#
# To add real Cloud-Optimized-GeoTIFF (COG) tiling (TiTiler/rio-tiler), build a
# heavier variant that also installs the GDAL-backed `[geo]` extra — see
# infra/Dockerfile.tiler.
# ---------------------------------------------------------------------------

# ---- builder: install the package + deps into an isolated venv -------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build into a self-contained virtualenv we can copy wholesale into the runtime
# stage (keeps the final image free of build tooling).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy only what the package build + runtime needs (keeps the layer cache warm
# and the context minimal). README.md is required because pyproject reads it as
# the long-description; configs/ + dashboard/ are served/read at runtime.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/

# Install the package with the slim, pure-wheel serving extra only.
# (For COG tiling add  ."[serving,geo]"  — requires system GDAL; see Dockerfile.tiler.)
RUN pip install --upgrade pip \
    && pip install ".[serving]"

# ---- runtime: minimal image with just the venv + app ----------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PORT=8080 \
    HOST=0.0.0.0

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    # Non-root runtime user (Cloud Run runs as the image user; least privilege).
    && groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Bring in the prebuilt venv and the runtime assets (code/configs/dashboard).
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/dashboard ./dashboard
COPY --from=builder /app/pyproject.toml /app/README.md ./

USER app

EXPOSE 8080

# Liveness probe (mirrors the Cloud Run startup/health check hitting /health).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Shell-form CMD so ${PORT} expands at container start. Cloud Run overrides
# $PORT at runtime; we bind 0.0.0.0:$PORT with a single Uvicorn worker because
# Cloud Run achieves concurrency by running multiple *instances*, not threads.
# To serve N workers per instance instead, swap for gunicorn, e.g.:
#   exec gunicorn agristress.serving.api:app -k uvicorn.workers.UvicornWorker \
#        -w "${WEB_CONCURRENCY:-1}" -b "0.0.0.0:${PORT}"
CMD exec uvicorn agristress.serving.api:app --host 0.0.0.0 --port "${PORT}" --workers 1

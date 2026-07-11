# ============================================================
# DOCKERFILE — House Price Prediction API
# ============================================================
# Multi-stage build:
#   Stage 1 (builder) — installs Python dependencies into a venv
#   Stage 2 (final)   — copies only the venv + app code, no build tools
# This keeps the final image small and free of compilers/caches.

# ── STAGE 1 — BUILDER ──────────────────────────────────────────
FROM python:3.13-slim AS builder

# WORKDIR sets the working directory for all following instructions.
# Also created automatically if it doesn't exist.
WORKDIR /app

# Create an isolated virtual environment inside the image.
# Even though the image itself is isolated, using a venv here lets us
# cleanly copy JUST the installed packages into the final stage below,
# without dragging along pip's cache or build metadata.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements.txt first (not the whole project).
# Docker caches each instruction as a layer. As long as requirements.txt
# doesn't change, Docker reuses this cached layer on rebuilds instead of
# reinstalling every dependency — this is what makes rebuilds fast when
# you're only changing app code.
COPY requirements.txt .

# --no-cache-dir prevents pip from storing its download cache in the
# image layer — that cache is only useful for repeated local installs,
# not inside a container that's built once and shipped.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── STAGE 2 — FINAL RUNTIME IMAGE ──────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy the fully-populated virtual environment from the builder stage.
# Nothing else from the builder (pip cache, apt lists, etc.) comes along.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PYTHONDONTWRITEBYTECODE — stops Python writing .pyc files into the
#   container filesystem (they're pointless in a short-lived container
#   and only add clutter/writes).
# PYTHONUNBUFFERED — forces stdout/stderr to be unbuffered, so your
#   print() and log statements show up immediately in `docker logs`
#   instead of being held in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a dedicated non-root user and group to run the app as.
# Running containers as root is an unnecessary security risk —
# if the process is ever compromised, a non-root user limits the damage.
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy only what the running API actually needs:
# app code and the trained model artifacts. Everything else
# (data/, notebook/, plots/, venv/, tests/, .git/) is excluded
# via .dockerignore and never enters the build context at all.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Hand ownership of the app directory to the non-root user
# (files are copied in as root by default, so this fixes permissions).
RUN chown -R appuser:appuser /app

# Switch to the non-root user for everything that follows,
# including the container's actual runtime process.
USER appuser

# Documents that the container listens on port 8000.
# This does NOT actually publish the port — that still requires
# `-p 8000:8000` (or docker-compose `ports:`) at run time.
# It's metadata for humans and tools like `docker inspect`.
EXPOSE 8000

# HEALTHCHECK lets Docker (and orchestrators like ECS/Kubernetes)
# know whether the container is actually serving traffic, not just
# "running". We hit the existing /health endpoint using Python's
# stdlib urllib, since curl isn't installed on the slim image and
# isn't worth adding just for this.
#   --interval    : how often to check
#   --timeout     : how long to wait for a response
#   --start-period: grace period after container start before
#                   failures count (model loading takes a moment)
#   --retries     : consecutive failures before marking "unhealthy"
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch the API using uvicorn against the app package (app.main:app),
# matching the standard production invocation — not `python app/main.py`.
# host=0.0.0.0 is required so the process accepts connections from
# outside the container, not just from within it.
# No --reload here: that flag is a dev-only convenience and has no
# place in a built image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

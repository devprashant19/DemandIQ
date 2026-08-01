# Multi-stage production container architecture for DemandIQ platform
# Stage 1: Build dependency wheelhouse and virtual environment
FROM python:3.11-slim-bookworm as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Install native build tools required for Prophet cmdstanpy compiler dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Stage 2: Runtime lean image with non-root security isolation
FROM python:3.11-slim-bookworm as runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install lightweight system requirements for runtime plotting and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash demandiq_user

COPY --from=builder /opt/venv /opt/venv

# Copy repository source code and assets
COPY --chown=demandiq_user:demandiq_user . /app/

# Create model and data persistence directories with proper read/write ownership
RUN mkdir -p /app/models /app/data/raw /app/data/processed /app/models/reports && \
    chown -R demandiq_user:demandiq_user /app

USER demandiq_user

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/demandiq/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# syntax=docker/dockerfile:1.7

FROM cgr.dev/chainguard/python:3.11-dev AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r /app/requirements.txt

FROM cgr.dev/chainguard/python:3.11 AS runtime

ARG APP_ENV=prd
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=${APP_ENV}

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src app models /app/
COPY configs/environments/base.yaml /app/configs/environments/base.yaml
COPY configs/environments/${APP_ENV}.yaml /app/configs/environments/${APP_ENV}.yaml

USER 65532:65532

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app/data \
    HF_HOME=/app/data/hf-cache

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Never COPY the repository root: it contains unreviewed source material.
COPY --chown=10001:10001 app/ /app/app/
COPY --chown=10001:10001 frontend/ /app/frontend/
RUN mkdir -p /app/data && chown 10001:10001 /app/data

USER 10001:10001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

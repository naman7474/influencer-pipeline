# Pipeline worker container.
# Build context = this repo root (the pipeline/ package itself).
#
#   docker build -t influencer-pipeline .
#   docker run --env-file .env -p 8000:8000 influencer-pipeline

FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies layer first so code changes don't invalidate the pip cache.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy everything into /app/pipeline so `pipeline.api` is importable.
COPY . /app/pipeline

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn pipeline.api:app --host 0.0.0.0 --port ${PORT}"]

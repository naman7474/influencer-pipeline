# Pipeline worker container.
# Build context is the repo root so we can COPY the whole `pipeline/` package.
#
#   docker build -f pipeline/Dockerfile -t influencer-pipeline .
#   docker run --env-file pipeline/.env -p 8000:8000 influencer-pipeline

FROM python:3.13-slim

# Whisper + BrightData clients need ffmpeg for any video processing that
# the CIP pipeline may do locally. Keep the image small by installing only
# what's required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies layer first so code changes don't invalidate the pip cache.
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/pipeline/requirements.txt

# Copy the pipeline package itself.
COPY pipeline /app/pipeline

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Simple healthcheck hits /health (no auth required).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# `sh -c` lets ${PORT} expand at runtime so hosts like Fly/Railway that
# inject their own PORT env var work without an override.
CMD ["sh", "-c", "uvicorn pipeline.api:app --host 0.0.0.0 --port ${PORT}"]

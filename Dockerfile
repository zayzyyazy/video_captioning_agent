# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRACK2_INPUT=/input/tasks.json \
    TRACK2_OUTPUT=/output/results.json \
    TRACK2_TEMP=/tmp/caption_agent

# Optional: bake credentials into the image for Track 2 evaluation.
# Prefer runtime -e FIREWORKS_API_KEY when possible.
ARG FIREWORKS_API_KEY=""
ARG FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
ARG FIREWORKS_VLM_MODEL="accounts/fireworks/models/kimi-k2p6"
ARG FIREWORKS_LLM_MODEL="accounts/fireworks/models/qwen3p7-plus"
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY} \
    FIREWORKS_BASE_URL=${FIREWORKS_BASE_URL} \
    FIREWORKS_VLM_MODEL=${FIREWORKS_VLM_MODEL} \
    FIREWORKS_LLM_MODEL=${FIREWORKS_LLM_MODEL}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

RUN mkdir -p /input /output /tmp/caption_agent

# Image architecture: linux/amd64, reads /input/tasks.json, writes /output/results.json, exits 0/1.
ENTRYPOINT ["python", "-m", "agent.main"]

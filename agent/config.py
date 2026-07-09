"""Runtime configuration for the captioning agent."""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


INPUT_PATH = Path(_env("TRACK2_INPUT", "/input/tasks.json"))
OUTPUT_PATH = Path(_env("TRACK2_OUTPUT", "/output/results.json"))
TEMP_DIR = Path(_env("TRACK2_TEMP", "/tmp/caption_agent"))

FIREWORKS_API_KEY = _env("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = _env(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
).rstrip("/")

# Vision model used to turn sampled frames into grounded observations.
# Prefer serverless VLMs that accept multiple images without a dedicated deployment.
FIREWORKS_VLM_MODEL = _env(
    "FIREWORKS_VLM_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct"
)
# Text model used to rewrite observations into the four required styles.
FIREWORKS_LLM_MODEL = _env(
    "FIREWORKS_LLM_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"
)

# Fallback chain if the primary model is unavailable / rate-limited.
VLM_FALLBACKS = [
    m.strip()
    for m in _env(
        "FIREWORKS_VLM_FALLBACKS",
        ",".join(
            [
                "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
                "accounts/fireworks/models/qwen2p5-vl-72b-instruct",
                "accounts/fireworks/models/llama4-maverick-instruct-basic",
                "accounts/fireworks/models/qwen2p5-vl-7b-instruct",
                "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
            ]
        ),
    ).split(",")
    if m.strip()
]

LLM_FALLBACKS = [
    m.strip()
    for m in _env(
        "FIREWORKS_LLM_FALLBACKS",
        ",".join(
            [
                "accounts/fireworks/models/llama-v3p3-70b-instruct",
                "accounts/fireworks/models/deepseek-v3p1",
                "accounts/fireworks/models/qwen3-235b-a22b",
                "accounts/fireworks/models/llama4-maverick-instruct-basic",
                "accounts/fireworks/models/mixtral-8x22b-instruct",
            ]
        ),
    ).split(",")
    if m.strip()
]

MAX_FRAMES = int(_env("MAX_FRAMES", "16"))
FRAME_WIDTH = int(_env("FRAME_WIDTH", "768"))
MAX_PARALLEL_TASKS = int(_env("MAX_PARALLEL_TASKS", "3"))
REQUEST_TIMEOUT = float(_env("REQUEST_TIMEOUT", "120"))
DOWNLOAD_TIMEOUT = float(_env("DOWNLOAD_TIMEOUT", "180"))
MAX_DOWNLOAD_MB = float(_env("MAX_DOWNLOAD_MB", "500"))

ALL_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

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
# Use current Fireworks *serverless* VLMs (older qwen2p5-vl-* IDs now 404).
FIREWORKS_VLM_MODEL = _env(
    "FIREWORKS_VLM_MODEL", "accounts/fireworks/models/kimi-k2p6"
)
# Text/style model. Multimodal serverless models also work well for caption rewrite.
FIREWORKS_LLM_MODEL = _env(
    "FIREWORKS_LLM_MODEL", "accounts/fireworks/models/qwen3p7-plus"
)

# Fallback chain if the primary model is unavailable / rate-limited.
VLM_FALLBACKS = [
    m.strip()
    for m in _env(
        "FIREWORKS_VLM_FALLBACKS",
        ",".join(
            [
                "accounts/fireworks/models/kimi-k2p6",
                "accounts/fireworks/models/qwen3p7-plus",
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/kimi-k2p7-code",
                "accounts/fireworks/models/kimi-k2p5",
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
                "accounts/fireworks/models/qwen3p7-plus",
                "accounts/fireworks/models/kimi-k2p6",
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/glm-5p2",
                "accounts/fireworks/models/deepseek-v3p2",
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

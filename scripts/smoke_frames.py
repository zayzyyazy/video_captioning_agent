#!/usr/bin/env python3
"""Smoke-test frame sampling without calling Fireworks."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.video import choose_timestamps, cleanup_task, prepare_frames  # noqa: E402


async def main() -> int:
    url = (
        "https://storage.googleapis.com/amd-hackathon-clips/"
        "13825391-uhd_3840_2160_30fps.mp4"
    )
    task_id = "smoke_v2"
    try:
        duration, frames = await prepare_frames(task_id, url)
    except Exception as exc:
        print(f"FAIL prepare_frames: {exc}", file=sys.stderr)
        return 1
    finally:
        # Keep frames briefly for size check then clean
        pass

    stamps = choose_timestamps(duration, 16)
    payload = {
        "duration": duration,
        "num_frames": len(frames),
        "timestamps": [round(t, 2) for t, _ in frames],
        "planned_timestamps": [round(t, 2) for t in stamps],
        "first_data_url_prefix": frames[0][1][:32] if frames else None,
    }
    print(json.dumps(payload, indent=2))
    cleanup_task(task_id)
    ok = duration > 0 and len(frames) >= 4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

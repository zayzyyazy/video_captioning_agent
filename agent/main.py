"""Entrypoint: read /input/tasks.json, write /output/results.json."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from agent import config
from agent.captioner import generate_style_captions, observe_video
from agent.fireworks import FireworksClient
from agent.video import cleanup_task, prepare_media

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("caption_agent")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tasks.json must be a JSON array")
    return data


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info("Wrote %d results to %s", len(results), path)


async def process_task(client: FireworksClient, task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id") or task.get("id") or "").strip()
    video_url = str(task.get("video_url") or "").strip()
    styles = task.get("styles") or list(config.ALL_STYLES)
    if not isinstance(styles, list) or not styles:
        styles = list(config.ALL_STYLES)
    styles = [str(s) for s in styles]

    if not task_id or not video_url:
        raise ValueError(f"Invalid task payload: {task}")

    logger.info("Processing task %s (%s)", task_id, video_url)
    try:
        media = await prepare_media(task_id, video_url)
        transcript = ""
        if media.audio_path is not None:
            transcript = await client.transcribe(media.audio_path)
            if transcript:
                logger.info(
                    "Task %s transcript ready (%d chars)", task_id, len(transcript)
                )
            else:
                logger.info("Task %s: no usable speech in audio", task_id)
        elif media.has_audio:
            logger.info("Task %s: audio present but extraction skipped/failed", task_id)
        else:
            logger.info("Task %s: no audio stream", task_id)

        observations = await observe_video(
            client,
            duration=media.duration,
            frames=media.frames,
            transcript=transcript,
        )
        logger.info("Task %s observations ready (%d chars)", task_id, len(observations))
        captions = await generate_style_captions(
            client,
            observations=observations,
            styles=styles,
            transcript=transcript,
        )
        # Ensure every requested style is present.
        for style in styles:
            captions.setdefault(style, f"A video scene described in {style} style.")
        return {"task_id": task_id, "captions": captions}
    finally:
        cleanup_task(task_id)


async def async_main() -> int:
    input_path = config.INPUT_PATH
    output_path = config.OUTPUT_PATH

    # Local convenience: fall back to ./sample_input if /input is absent.
    if not input_path.exists():
        alt = Path("sample_input/tasks.json")
        if alt.exists():
            logger.warning("%s missing; using %s", input_path, alt)
            input_path = alt
            if str(output_path) == "/output/results.json":
                output_path = Path("sample_output/results.json")

    try:
        tasks = load_tasks(input_path)
    except Exception as exc:
        logger.error("Failed to load tasks: %s", exc)
        return 1

    if not tasks:
        logger.error("No tasks found in %s", input_path)
        return 1

    try:
        client = FireworksClient()
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    sem = asyncio.Semaphore(max(1, config.MAX_PARALLEL_TASKS))
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    errors: list[str] = []

    async def _run(idx: int, task: dict[str, Any]) -> None:
        async with sem:
            try:
                results[idx] = await process_task(client, task)
            except Exception as exc:
                tid = task.get("task_id", idx)
                msg = f"Task {tid} failed: {exc}"
                logger.error(msg)
                traceback.print_exc()
                errors.append(msg)
                results[idx] = None

    try:
        await asyncio.gather(*[_run(i, t) for i, t in enumerate(tasks)])
    finally:
        await client.aclose()

    final: list[dict[str, Any]] = []
    for item in results:
        if not item:
            continue
        clean = {
            "task_id": item["task_id"],
            "captions": item["captions"],
        }
        final.append(clean)

    try:
        write_results(output_path, final)
    except Exception as exc:
        logger.error("Failed to write results: %s", exc)
        return 1

    if errors:
        logger.error("%d task(s) failed", len(errors))
        return 1
    logger.info("All %d task(s) completed successfully", len(final))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

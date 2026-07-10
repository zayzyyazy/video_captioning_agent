"""Video download, metadata, frame sampling, and audio extraction via ffmpeg."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from agent import config

logger = logging.getLogger(__name__)


@dataclass
class PreparedMedia:
    duration: float
    frames: list[tuple[float, str]]
    audio_path: Path | None
    has_audio: bool
    work_dir: Path | None = None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_media(video_path: Path) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio_stream)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[-400:]}")
    data = json.loads(result.stdout or "{}")
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration")
    has_audio = any(
        stream.get("codec_type") == "audio" for stream in data.get("streams", [])
    )
    return duration, has_audio


def choose_timestamps(duration: float, max_frames: int) -> list[float]:
    """Evenly sample across the full clip, avoiding exact start/end edges."""
    n = max(1, min(max_frames, max(4, int(math.ceil(duration / 5.0)))))
    if duration <= 1.0:
        return [max(0.0, duration * 0.5)]
    # Keep a small margin so we don't land on black intro/outro frames.
    margin = min(0.35, duration * 0.02)
    start = margin
    end = max(margin, duration - margin)
    if n == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def extract_frame_jpeg(video_path: Path, timestamp: float, out_path: Path, width: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "3",
        str(out_path),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg frame extract failed at {timestamp:.2f}s: {result.stderr[-400:]}"
        )
    return out_path


def extract_audio_mp3(video_path: Path, out_path: Path) -> Path | None:
    """Extract mono 16kHz MP3 for Whisper. Returns None if extraction fails/empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(out_path),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 256:
        logger.warning(
            "Audio extraction failed or empty: %s",
            (result.stderr or "")[-300:],
        )
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    logger.info(
        "Extracted audio %s (%.1f KB)",
        out_path.name,
        out_path.stat().st_size / 1024.0,
    )
    return out_path


def jpeg_to_data_url(path: Path, max_side: int = 768, quality: int = 85) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def download_video(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(config.DOWNLOAD_TIMEOUT, connect=30.0)
    max_bytes = int(config.MAX_DOWNLOAD_MB * 1024 * 1024)
    tmp_dest = dest.with_suffix(dest.suffix + ".partial")
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = 0
            with open(tmp_dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(
                            f"Download exceeded {config.MAX_DOWNLOAD_MB} MB limit"
                        )
                    f.write(chunk)
    tmp_dest.replace(dest)
    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / (1024 * 1024))
    return dest


def work_dir_for(task_id: str, unique_key: str) -> Path:
    """Unique scratch dir so duplicate task_ids never collide under concurrency."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:64]
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in unique_key)[:80]
    return config.TEMP_DIR / f"{safe_id}__{safe_key}"


def _fit_frames_under_budget(
    frames: list[tuple[float, str]],
    budget_bytes: int = 8_500_000,
) -> list[tuple[float, str]]:
    """Drop frames from the middle outward until base64 payload fits Fireworks limits."""
    if not frames:
        return frames

    def total_size(items: list[tuple[float, str]]) -> int:
        return sum(len(url) for _, url in items)

    selected = list(frames)
    while len(selected) > 4 and total_size(selected) > budget_bytes:
        mid = len(selected) // 2
        selected.pop(mid)
    if total_size(selected) > budget_bytes and len(selected) > 2:
        mid = len(frames) // 2
        selected = [frames[0], frames[mid], frames[-1]]
    logger.info(
        "Frame payload ~%.1f MB across %d frames",
        total_size(selected) / (1024 * 1024),
        len(selected),
    )
    return selected


async def prepare_media(
    task_id: str, video_url: str, *, unique_key: str | None = None
) -> PreparedMedia:
    """Download video, sample frames, and optionally extract audio for Whisper."""
    key = unique_key or task_id
    work = work_dir_for(task_id, key)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    video_path = work / "clip.mp4"
    await download_video(video_url, video_path)

    duration, has_audio = await asyncio.to_thread(probe_media, video_path)
    timestamps = choose_timestamps(duration, config.MAX_FRAMES)
    logger.info(
        "Task %s: duration=%.1fs has_audio=%s sampling %d frames",
        task_id,
        duration,
        has_audio,
        len(timestamps),
    )

    audio_path: Path | None = None
    if has_audio and config.ENABLE_TRANSCRIPTION:
        candidate = work / "audio.mp3"
        audio_path = await asyncio.to_thread(extract_audio_mp3, video_path, candidate)

    frames: list[tuple[float, str]] = []
    for idx, ts in enumerate(timestamps):
        frame_path = work / f"frame_{idx:02d}.jpg"
        await asyncio.to_thread(
            extract_frame_jpeg, video_path, ts, frame_path, config.FRAME_WIDTH
        )
        data_url = await asyncio.to_thread(
            jpeg_to_data_url, frame_path, config.FRAME_WIDTH
        )
        frames.append((ts, data_url))

    frames = _fit_frames_under_budget(frames)

    # Free the large video file; keep frames + audio until task completes.
    try:
        video_path.unlink(missing_ok=True)
    except OSError:
        pass

    return PreparedMedia(
        duration=duration,
        frames=frames,
        audio_path=audio_path,
        has_audio=has_audio,
        work_dir=work,
    )


# Back-compat alias used by older smoke scripts.
async def prepare_frames(
    task_id: str, video_url: str
) -> tuple[float, list[tuple[float, str]]]:
    media = await prepare_media(task_id, video_url)
    return media.duration, media.frames


def cleanup_work(work_dir: Path | None) -> None:
    if work_dir is None:
        return
    shutil.rmtree(work_dir, ignore_errors=True)


def cleanup_task(task_id: str) -> None:
    # Legacy helper: remove any dirs that start with the task id prefix.
    root = config.TEMP_DIR
    if not root.exists():
        return
    for path in root.glob(f"{task_id}*"):
        shutil.rmtree(path, ignore_errors=True)

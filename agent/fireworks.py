"""Fireworks OpenAI-compatible chat + Whisper transcription client."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from agent import config

logger = logging.getLogger(__name__)


class FireworksError(RuntimeError):
    pass


def _unique(models: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _strip_thinking(text: str) -> str:
    """Remove reasoning / think blocks that some Fireworks models emit."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<reason>[\s\S]*?</reason>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_json(text: str) -> Any:
    text = _strip_thinking(text)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Prefer the last JSON object in the string (after any preamble).
        matches = list(re.finditer(r"\{[\s\S]*\}", text))
        if not matches:
            raise
        last_error: Exception | None = None
        for match in reversed(matches):
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                # Try a tighter object around "captions"
                cap = re.search(r'\{\s*"captions"\s*:\s*\{[\s\S]*?\}\s*\}', candidate)
                if cap:
                    try:
                        return json.loads(cap.group(0))
                    except json.JSONDecodeError as exc2:
                        last_error = exc2
        if last_error:
            raise last_error
        raise


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "output_text") and part.get("text"):
                    parts.append(str(part["text"]))
                elif "text" in part:
                    parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        content = "".join(parts)
    text = (content or "").strip()
    if not text:
        # Some reasoning models put the final answer elsewhere.
        for key in ("reasoning_content", "reasoning"):
            alt = message.get(key)
            if isinstance(alt, str) and alt.strip():
                text = alt.strip()
                break
    return text


class FireworksClient:
    def __init__(self) -> None:
        if not config.FIREWORKS_API_KEY:
            raise FireworksError(
                "FIREWORKS_API_KEY is missing. Pass it as an env var or Docker build-arg."
            )
        self.base_url = config.FIREWORKS_BASE_URL
        self.api_key = config.FIREWORKS_API_KEY
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(config.REQUEST_TIMEOUT, connect=30.0),
        )
        # Separate client for Whisper (different host, multipart uploads).
        self._audio_client = httpx.AsyncClient(
            base_url=config.FIREWORKS_AUDIO_BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(config.TRANSCRIBE_TIMEOUT, connect=30.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._audio_client.aclose()

    async def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio with Fireworks Whisper. Returns empty string on soft failure."""
        if not config.ENABLE_TRANSCRIPTION:
            return ""
        if not audio_path or not audio_path.exists():
            return ""

        # (base_url, model) pairs — turbo first for speed, prod as fallback.
        endpoints: list[tuple[str, str]] = [
            (config.FIREWORKS_AUDIO_BASE_URL, config.FIREWORKS_WHISPER_MODEL),
            (
                "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1",
                "whisper-v3-turbo",
            ),
            (
                "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1",
                "whisper-v3",
            ),
            # Some accounts expose Whisper on the main inference host.
            ("https://api.fireworks.ai/inference/v1", "whisper-v3-turbo"),
            ("https://api.fireworks.ai/inference/v1", "whisper-v3"),
        ]
        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[str, str]] = []
        for pair in endpoints:
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

        # Fireworks audio docs/examples vary between raw key and Bearer.
        auth_headers = [
            {"Authorization": f"Bearer {self.api_key}"},
            {"Authorization": self.api_key},
        ]

        last_error: Exception | None = None
        audio_bytes = audio_path.read_bytes()
        filename = audio_path.name or "audio.mp3"

        for base_url, model in pairs:
            for auth in auth_headers:
                for attempt in range(1, 3):
                    try:
                        files = {"file": (filename, audio_bytes, "audio/mpeg")}
                        data = {
                            "model": model,
                            "response_format": "json",
                            "temperature": "0.0",
                        }
                        resp = await self._audio_client.post(
                            f"{base_url}/audio/transcriptions",
                            data=data,
                            files=files,
                            headers=auth,
                        )
                        if resp.status_code in (429, 500, 502, 503, 504):
                            wait = min(2 ** attempt, 12)
                            logger.warning(
                                "Whisper %s returned %s; retry in %ss",
                                model,
                                resp.status_code,
                                wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        if resp.status_code in (401, 403):
                            last_error = FireworksError(
                                f"Whisper {model} HTTP {resp.status_code}: {resp.text[:200]}"
                            )
                            # Try next auth style / endpoint.
                            break
                        if resp.status_code == 404:
                            logger.warning(
                                "Whisper unavailable at %s model=%s", base_url, model
                            )
                            break
                        if resp.status_code >= 400:
                            body = resp.text[:400]
                            last_error = FireworksError(
                                f"Whisper {model} HTTP {resp.status_code}: {body}"
                            )
                            logger.warning("%s", last_error)
                            break

                        payload = resp.json()
                        if isinstance(payload, dict):
                            text = str(payload.get("text") or "").strip()
                        else:
                            text = str(payload).strip()
                        text = re.sub(r"\s+", " ", text).strip()
                        if text:
                            logger.info(
                                "Whisper success via %s (%d chars)", model, len(text)
                            )
                            if len(text) > config.MAX_TRANSCRIPT_CHARS:
                                text = (
                                    text[: config.MAX_TRANSCRIPT_CHARS].rstrip() + "…"
                                )
                            return text
                        logger.info(
                            "Whisper %s returned empty transcript (likely no speech)",
                            model,
                        )
                        return ""
                    except (httpx.HTTPError, FireworksError, ValueError, KeyError) as exc:
                        last_error = exc
                        wait = min(2 ** attempt, 12)
                        logger.warning(
                            "Whisper call failed (%s, attempt %s): %s",
                            model,
                            attempt,
                            exc,
                        )
                        await asyncio.sleep(wait)

        logger.warning(
            "Transcription unavailable (vision captions still run). Last error: %s",
            last_error,
        )
        return ""

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        models: list[str],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        expect_json: bool = False,
        disable_reasoning: bool = True,
    ) -> str:
        last_error: Exception | None = None
        for model in _unique(models):
            use_json_format = expect_json
            use_reasoning_none = disable_reasoning
            for attempt in range(1, 4):
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if use_json_format:
                    payload["response_format"] = {"type": "json_object"}
                if use_reasoning_none:
                    # Disable thinking traces on Qwen/Kimi-style Fireworks models.
                    payload["reasoning_effort"] = "none"
                try:
                    resp = await self._client.post("/chat/completions", json=payload)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        wait = min(2 ** attempt, 12)
                        logger.warning(
                            "Fireworks %s returned %s for %s; retry in %ss",
                            model,
                            resp.status_code,
                            attempt,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code == 404:
                        body = resp.text[:300]
                        last_error = FireworksError(
                            f"{model} HTTP 404 (not on serverless / wrong id): {body}"
                        )
                        logger.warning("%s", last_error)
                        break
                    if resp.status_code >= 400:
                        body = resp.text[:500]
                        lower = body.lower()
                        # Drop unsupported request fields and retry.
                        if use_json_format and "response_format" in lower:
                            use_json_format = False
                            continue
                        if use_reasoning_none and (
                            "reasoning_effort" in lower or "reasoning" in lower
                        ):
                            use_reasoning_none = False
                            continue
                        last_error = FireworksError(
                            f"{model} HTTP {resp.status_code}: {body}"
                        )
                        raise last_error
                    data = resp.json()
                    message = data["choices"][0]["message"]
                    text = _message_text(message)
                    text = _strip_thinking(text)
                    if not text:
                        raise FireworksError(f"{model} returned empty content")
                    logger.info("Fireworks success via %s", model)
                    return text
                except (httpx.HTTPError, FireworksError, KeyError, IndexError) as exc:
                    last_error = exc
                    wait = min(2 ** attempt, 12)
                    logger.warning(
                        "Fireworks call failed (%s, attempt %s): %s",
                        model,
                        attempt,
                        exc,
                    )
                    await asyncio.sleep(wait)
        raise FireworksError(f"All Fireworks models failed: {last_error}")

    async def chat_json(
        self,
        *,
        messages: list[dict[str, Any]],
        models: list[str],
        temperature: float = 0.4,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        text = await self.chat(
            messages=messages,
            models=models,
            temperature=temperature,
            max_tokens=max_tokens,
            expect_json=True,
            disable_reasoning=True,
        )
        data = _extract_json(text)
        if not isinstance(data, dict):
            raise FireworksError(f"Expected JSON object, got {type(data)}")
        return data

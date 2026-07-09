"""Fireworks OpenAI-compatible chat client with retries and model fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

    async def aclose(self) -> None:
        await self._client.aclose()

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

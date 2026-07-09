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


def _extract_json(text: str) -> Any:
    text = text.strip()
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
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


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
    ) -> str:
        last_error: Exception | None = None
        for model in _unique(models):
            for attempt in range(1, 4):
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if expect_json:
                    payload["response_format"] = {"type": "json_object"}
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
                        logger.warning("Model unavailable: %s", model)
                        break
                    if resp.status_code >= 400:
                        body = resp.text[:500]
                        # Some models reject response_format; retry without it.
                        if expect_json and "response_format" in body.lower():
                            expect_json = False
                            continue
                        raise FireworksError(
                            f"{model} HTTP {resp.status_code}: {body}"
                        )
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    text = (content or "").strip()
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
        )
        data = _extract_json(text)
        if not isinstance(data, dict):
            raise FireworksError(f"Expected JSON object, got {type(data)}")
        return data

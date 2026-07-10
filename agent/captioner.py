"""Observation + styled caption generation via Fireworks."""

from __future__ import annotations

import logging
import re
from typing import Any

from agent import config, prompts
from agent.fireworks import FireworksClient, FireworksError

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(
    r"^(?:\.\.\.|…|caption here|todo|tbd|n/?a|none|null|-)?$",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.replace("—", " - ").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _is_placeholder(value: str) -> bool:
    v = value.strip()
    if len(v) < 12:
        return True
    if _PLACEHOLDER_RE.match(v):
        return True
    if v in {"...", "…"}:
        return True
    # Reject captions that are just the sample examples from the prompt.
    sample_bits = (
        "orange kitten sits among green garden foliage",
        "kitten on leaf patrol",
        "process kitten.exe",
        "tiny orange floof",
    )
    low = v.lower()
    return any(bit in low for bit in sample_bits)


def _pick_captions(data: dict[str, Any], styles: list[str]) -> dict[str, str]:
    raw = data.get("captions") if isinstance(data.get("captions"), dict) else data
    if not isinstance(raw, dict):
        raise FireworksError("Caption JSON missing captions object")

    normalized = {_normalize_key(k): _clean(str(v)) for k, v in raw.items()}
    out: dict[str, str] = {}
    for style in styles:
        key = _normalize_key(style)
        value = normalized.get(key, "")
        if not value:
            for nk, nv in normalized.items():
                if key in nk or nk in key:
                    value = nv
                    break
        if value and not _is_placeholder(value):
            out[style] = value
        elif value:
            logger.warning("Rejecting placeholder caption for %s: %r", style, value)
    return out


def _context_block(observations: str, transcript: str | None) -> str:
    """Merge visual observations + transcript for captioning / fallbacks."""
    obs = _clean(observations)
    tx = (transcript or "").strip()
    if tx:
        return f"{obs} SPEECH/TRANSCRIPT: {tx}"
    return obs


async def observe_video(
    client: FireworksClient,
    *,
    duration: float,
    frames: list[tuple[float, str]],
    transcript: str | None = None,
) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompts.OBSERVATION_USER.format(
                duration_hint=f"about {duration:.0f} seconds long",
                transcript=prompts.format_transcript(transcript),
            ),
        }
    ]
    # Keep payload under Fireworks' ~10MB / 30-image limits.
    for ts, data_url in frames[: config.MAX_FRAMES]:
        content.append({"type": "text", "text": f"Frame at t={ts:.1f}s:"})
        content.append(
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
        )

    messages = [
        {"role": "system", "content": prompts.OBSERVATION_SYSTEM},
        {"role": "user", "content": content},
    ]
    models = [config.FIREWORKS_VLM_MODEL, *config.VLM_FALLBACKS]
    text = await client.chat(
        messages=messages,
        models=models,
        temperature=0.1,
        max_tokens=1000,
        expect_json=False,
        disable_reasoning=True,
    )
    return _clean(text)


async def generate_style_captions(
    client: FireworksClient,
    *,
    observations: str,
    styles: list[str],
    transcript: str | None = None,
) -> dict[str, str]:
    styles = [s for s in styles if s]
    if not styles:
        styles = list(config.ALL_STYLES)

    transcript_text = prompts.format_transcript(transcript)
    user = prompts.CAPTION_USER.format(
        observations=observations,
        transcript=transcript_text,
        styles=", ".join(styles),
        style_block=prompts.style_block(styles),
        example_json=prompts.example_json(styles),
    )
    messages = [
        {"role": "system", "content": prompts.CAPTION_SYSTEM},
        {"role": "user", "content": user},
    ]
    models = [config.FIREWORKS_LLM_MODEL, *config.LLM_FALLBACKS]

    try:
        data = await client.chat_json(
            messages=messages,
            models=models,
            temperature=0.55,
            max_tokens=1200,
        )
        captions = _pick_captions(data, styles)
    except Exception as exc:
        logger.warning("Primary caption generation failed: %s", exc)
        captions = {}

    missing = [s for s in styles if not captions.get(s)]
    if missing:
        for style in missing:
            single_user = prompts.CAPTION_USER.format(
                observations=observations,
                transcript=transcript_text,
                styles=style,
                style_block=prompts.style_block([style]),
                example_json=prompts.example_json([style]),
            )
            try:
                data = await client.chat_json(
                    messages=[
                        {"role": "system", "content": prompts.CAPTION_SYSTEM},
                        {"role": "user", "content": single_user},
                    ],
                    models=models + config.VLM_FALLBACKS,
                    temperature=0.5,
                    max_tokens=500,
                )
                got = _pick_captions(data, [style])
                if got.get(style):
                    captions[style] = got[style]
            except Exception as exc:
                logger.warning("Per-style caption failed for %s: %s", style, exc)

    context = _context_block(observations, transcript)
    for style in styles:
        if not captions.get(style) or _is_placeholder(captions[style]):
            captions[style] = _fallback_caption(style, context)

    return {s: captions[s] for s in styles}


def _fallback_caption(style: str, observations: str) -> str:
    snippet = observations[:280].rstrip(" .")
    if style == "formal":
        return f"The video shows the following scene: {snippet}."
    if style == "sarcastic":
        return f"Sure, nothing to see here - just {snippet.lower()}."
    if style == "humorous_tech":
        return (
            f"Runtime log: scene.render() succeeded with payload "
            f'"{snippet[:160]}".'
        )
    return f"Okay, picture this: {snippet}."

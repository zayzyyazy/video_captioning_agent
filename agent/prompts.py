"""Prompt templates for observation and styled caption generation."""

from __future__ import annotations

OBSERVATION_SYSTEM = """You are a careful video analyst. You receive ordered still frames sampled across a video clip.
Describe ONLY what is visibly present. Do not invent people, objects, brands, weather, or actions that are not supported by the frames.
Be concrete about setting, subjects, clothing, objects, actions, lighting, weather, and any readable on-screen text.
Write a neutral factual brief that another writer can turn into captions."""

OBSERVATION_USER = """These frames are in chronological order from a video clip ({duration_hint}).
Each image is labeled with its approximate timestamp.

Produce a detailed factual observation brief with these sections:
SETTING: location / environment / time-of-day / weather / lighting
SUBJECTS: people, animals, or main subjects (appearance, clothing, notable traits)
OBJECTS: important props, vehicles, devices, food, signage, etc.
ACTIONS: what happens over time across the frames (chronological)
VISIBLE TEXT: any readable text/logos/signs (quote exactly); otherwise "none"
UNCERTAINTIES: anything ambiguous; otherwise "none"

Keep the brief under 250 words. Do not write captions or jokes."""

CAPTION_SYSTEM = """You write short video captions that stay faithful to a factual observation brief.
Never mention frames, OCR, models, AI, pipelines, timestamps, or analysis.
Never invent details that are not in the brief.
Output valid JSON only."""

STYLE_GUIDE = {
    "formal": (
        "Professional, objective, factual tone. Describe only observable events. "
        "No jokes, no irony, no slang, no invented emotion."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking. Target ordinary situations or obvious contradictions "
        "in the scene. Keep sarcasm subtle and conversational; do not invent unrelated jokes."
    ),
    "humorous_tech": (
        "Funny with technology or programming references. Describe the scene through ONE "
        "consistent tech metaphor (OS, game engine, API, debugger, robotics, networking). "
        "Stay grounded in the actual scene."
    ),
    "humorous_non_tech": (
        "Funny everyday humour with NO technical jargon. Observational comedy a witty friend "
        "would use. No sarcasm-heavy tone and no programming metaphors."
    ),
}

CAPTION_USER = """Factual observation brief:
{observations}

Write one caption (1-3 sentences) for EACH requested style.
Requested styles: {styles}

Style requirements:
{style_block}

Rules:
1. Every caption must reflect the same real scene from the brief.
2. If visible text is present, weave it in naturally when relevant.
3. Keep humour lighthearted; no insults, politics, or sexual content.
4. Return ONLY JSON of the form:
{{"captions": {{{style_json_keys}}}}}
"""


def style_block(styles: list[str]) -> str:
    lines = []
    for style in styles:
        guide = STYLE_GUIDE.get(style, "Match the requested tone closely.")
        lines.append(f"- {style}: {guide}")
    return "\n".join(lines)


def style_json_keys(styles: list[str]) -> str:
    return ", ".join(f'"{s}": "..."' for s in styles)

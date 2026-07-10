"""Prompt templates for observation and styled caption generation."""

from __future__ import annotations

import json

OBSERVATION_SYSTEM = """You are a careful video analyst. You receive ordered still frames sampled across a video clip, plus an optional speech/audio transcript.
Ground your brief in BOTH the visuals and the transcript when speech is present.
Do not invent people, objects, brands, weather, dialogue, or actions that are not supported by the frames or transcript.
Be concrete about setting, subjects, clothing, objects, actions, lighting, weather, readable on-screen text, and spoken content.
Write a neutral factual brief that another writer can turn into captions."""

OBSERVATION_USER = """These frames are in chronological order from a video clip ({duration_hint}).
Each image is labeled with its approximate timestamp.

Audio transcript (may be empty if the clip has no speech):
{transcript}

Produce a detailed factual observation brief with these sections:
SETTING: location / environment / time-of-day / weather / lighting
SUBJECTS: people, animals, or main subjects (appearance, clothing, notable traits)
OBJECTS: important props, vehicles, devices, food, signage, etc.
ACTIONS: what happens over time across the frames (chronological)
SPEECH: summarize or quote key spoken content from the transcript; otherwise "none"
VISIBLE TEXT: any readable text/logos/signs (quote exactly); otherwise "none"
UNCERTAINTIES: anything ambiguous; otherwise "none"

Keep the brief under 280 words. Do not write captions or jokes.
If the transcript conflicts with visuals, prefer what is clearly visible and note the conflict under UNCERTAINTIES."""

CAPTION_SYSTEM = """You write short video captions that stay faithful to a factual observation brief and optional audio transcript.
Never mention frames, OCR, models, AI, pipelines, timestamps, transcripts, or analysis.
Never invent details that are not in the brief/transcript.
Never use ellipsis placeholders. Every caption must be a complete natural-language sentence.
Return a single JSON object only. No markdown fences. No thinking aloud."""

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

Audio transcript (may be empty):
{transcript}

Write one caption (1-3 complete sentences) for EACH requested style.
Requested styles: {styles}

Style requirements:
{style_block}

Rules:
1. Every caption must reflect the same real scene from the brief (and speech when relevant).
2. If visible text or spoken lines matter, weave them in naturally.
3. Keep humour lighthearted; no insults, politics, or sexual content.
4. Do NOT copy placeholders. Do NOT output "...", "caption here", or empty strings.
5. Return ONLY a JSON object with this shape (replace each value with a real caption):
{example_json}
"""


def style_block(styles: list[str]) -> str:
    lines = []
    for style in styles:
        guide = STYLE_GUIDE.get(style, "Match the requested tone closely.")
        lines.append(f"- {style}: {guide}")
    return "\n".join(lines)


def example_json(styles: list[str]) -> str:
    # Concrete sample values so the model does not copy "..."
    samples = {
        "formal": "An orange kitten sits among green garden foliage in daylight.",
        "sarcastic": "Clearly the garden's most productive employee: a kitten on leaf patrol.",
        "humorous_tech": "Process kitten.exe spawned in GardenOS with foliage shaders at max.",
        "humorous_non_tech": "This tiny orange floof is out here living its best leafy life.",
    }
    captions = {
        style: samples.get(style, f"A real {style} caption about the scene.")
        for style in styles
    }
    return json.dumps({"captions": captions}, ensure_ascii=False)


def format_transcript(transcript: str | None) -> str:
    text = (transcript or "").strip()
    if not text:
        return "No speech detected / no audio transcript available."
    return text

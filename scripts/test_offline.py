#!/usr/bin/env python3
"""Offline unit checks (no Fireworks calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.captioner import _fallback_caption, _pick_captions  # noqa: E402
from agent.fireworks import _extract_json  # noqa: E402
from agent.video import choose_timestamps  # noqa: E402


def test_timestamps() -> None:
    ts = choose_timestamps(60.0, 16)
    assert 4 <= len(ts) <= 16
    assert ts[0] > 0
    assert ts[-1] < 60.0
    assert ts == sorted(ts)


def test_json_extract() -> None:
    raw = '```json\n{"captions": {"formal": "A cat."}}\n```'
    data = _extract_json(raw)
    assert data["captions"]["formal"] == "A cat."


def test_pick_captions() -> None:
    data = {
        "captions": {
            "Formal": "One",
            "humorous-tech": "Two",
            "sarcastic": "Three",
            "humorous_non_tech": "Four",
        }
    }
    styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    out = _pick_captions(data, styles)
    assert out["formal"] == "One"
    assert out["humorous_tech"] == "Two"
    assert out["sarcastic"] == "Three"
    assert out["humorous_non_tech"] == "Four"


def test_fallback() -> None:
    obs = "SETTING: garden. SUBJECTS: orange kitten."
    for style in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        cap = _fallback_caption(style, obs)
        assert len(cap) > 10


if __name__ == "__main__":
    test_timestamps()
    test_json_extract()
    test_pick_captions()
    test_fallback()
    print("all unit tests passed")

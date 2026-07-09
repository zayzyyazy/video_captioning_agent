#!/usr/bin/env python3
"""Offline unit checks (no Fireworks calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.captioner import (  # noqa: E402
    _fallback_caption,
    _is_placeholder,
    _pick_captions,
)
from agent.fireworks import _extract_json  # noqa: E402
from agent.prompts import example_json  # noqa: E402
from agent.video import choose_timestamps  # noqa: E402


def test_timestamps() -> None:
    ts = choose_timestamps(60.0, 16)
    assert 4 <= len(ts) <= 16
    assert ts[0] > 0
    assert ts[-1] < 60.0
    assert ts == sorted(ts)


def test_json_extract() -> None:
    raw = '```json\n{"captions": {"formal": "A fluffy orange cat in a garden."}}\n```'
    data = _extract_json(raw)
    assert "fluffy orange cat" in data["captions"]["formal"]

    thinking = (
        '<think>planning</think>\n'
        '{"captions": {"formal": "A fluffy orange cat in a garden."}}'
    )
    data2 = _extract_json(thinking)
    assert "fluffy orange cat" in data2["captions"]["formal"]


def test_pick_captions() -> None:
    data = {
        "captions": {
            "Formal": "A woman works at a desk in a bright office.",
            "humorous-tech": "User desk_worker.exe is stuck in an infinite mouse loop.",
            "sarcastic": "Another thrilling day of moving a mouse under fluorescent lights.",
            "humorous_non_tech": "She is giving that office plant a masterclass in looking busy.",
        }
    }
    styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    out = _pick_captions(data, styles)
    assert "bright office" in out["formal"]
    assert "mouse loop" in out["humorous_tech"]
    assert "fluorescent" in out["sarcastic"]
    assert "office plant" in out["humorous_non_tech"]


def test_reject_placeholders() -> None:
    assert _is_placeholder("...")
    assert _is_placeholder("…")
    assert _is_placeholder("todo")
    data = {"captions": {"formal": "...", "sarcastic": "A dry take on office life today."}}
    out = _pick_captions(data, ["formal", "sarcastic"])
    assert "formal" not in out
    assert "sarcastic" in out


def test_example_json() -> None:
    raw = example_json(["formal"])
    assert "..." not in raw
    assert "captions" in raw


def test_fallback() -> None:
    obs = "SETTING: garden. SUBJECTS: orange kitten."
    for style in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        cap = _fallback_caption(style, obs)
        assert len(cap) > 10
        assert not _is_placeholder(cap)


if __name__ == "__main__":
    test_timestamps()
    test_json_extract()
    test_pick_captions()
    test_reject_placeholders()
    test_example_json()
    test_fallback()
    print("all unit tests passed")

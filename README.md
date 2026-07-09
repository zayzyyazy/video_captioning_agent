# Track 2: Video Captioning Agent (Fireworks only)

Dockerized agent for the AMD Developer Hackathon ACT II — Track 2.

It reads `/input/tasks.json`, downloads each clip, samples timeline frames with
ffmpeg, asks a Fireworks vision model for grounded observations, then asks a
Fireworks text model for captions in every requested style, and writes
`/output/results.json`.

## Pipeline

```text
tasks.json
  -> download mp4
  -> ffprobe duration
  -> ffmpeg evenly spaced JPEG frames (adaptive count, capped)
  -> Fireworks VLM observations (setting/subjects/objects/actions/text)
  -> Fireworks LLM styled captions (formal / sarcastic / humorous_tech / humorous_non_tech)
  -> /output/results.json
```

## Styles

| Style | Tone |
| --- | --- |
| `formal` | Professional, objective, factual |
| `sarcastic` | Dry, ironic, lightly mocking |
| `humorous_tech` | Funny with tech/programming metaphors |
| `humorous_non_tech` | Everyday humour, no jargon |

## Requirements

- Docker (`linux/amd64`)
- A Fireworks API key (`FIREWORKS_API_KEY`)

Track 2 does **not** inject API keys. Bake the key into the image with a build
arg, or pass it at runtime with `-e`.

## Local run (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FIREWORKS_API_KEY=fw_...
export TRACK2_INPUT=sample_input/tasks.json
export TRACK2_OUTPUT=sample_output/results.json
python -m agent.main
```

## Docker

### Build (bake key for submission)

```bash
docker buildx build --platform linux/amd64 \
  --build-arg FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -t ghcr.io/zayzyyazy/video_captioning_agent:latest \
  --push .
```

### Run like the evaluation harness

```bash
mkdir -p input output
cp sample_input/tasks.json input/tasks.json

docker run --rm \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  ghcr.io/zayzyyazy/video_captioning_agent:latest
```

Exit code `0` on full success, non-zero if any task fails.

## Image architecture

- Platform: `linux/amd64`
- Entrypoint: `python -m agent.main`
- Reads: `/input/tasks.json`
- Writes: `/output/results.json`
- Compressed size target: well under 10GB (slim Python + ffmpeg)
- Max runtime budget: 10 minutes (parallel tasks + capped frames)

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIREWORKS_API_KEY` | _(required)_ | Fireworks credential |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | API base |
| `FIREWORKS_VLM_MODEL` | `accounts/fireworks/models/qwen2p5-vl-32b-instruct` | Vision model |
| `FIREWORKS_LLM_MODEL` | `accounts/fireworks/models/llama-v3p3-70b-instruct` | Caption model |
| `MAX_FRAMES` | `16` | Frame sample cap |
| `MAX_PARALLEL_TASKS` | `3` | Concurrent clips |

## Output schema

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

# Docker / OVOS TTS Server

This page is for operators running phoonnx as a containerized TTS server. It covers the
published image, its configuration, and how to build it locally.

phoonnx ships a **batteries-included** image that runs it as an
[`ovos-tts-server`](https://github.com/OpenVoiceOS/ovos-tts-server) — with **every
optional dependency** (all language phonemizers + voice cloning), the **espeak-ng**
binary, and a **pre-filled voice index** so it starts cleanly on a cold cache.

## Quick start

```bash
docker run -p 9666:9666 ghcr.io/tigregotico/phoonnx:latest
```

or with the bundled compose file:

```bash
docker compose up -d
```

Then synthesize:

```bash
curl "http://localhost:9666/synthesize/hello%20world" --output hello.wav
# status check
curl http://localhost:9666/status
```

## What's in the image

| | |
|---|---|
| Base | `python:3.11-slim` |
| System | `espeak-ng`, `libsndfile1` |
| Python | `phoonnx[all]` (every language extra + `cloning`) + `ovos-tts-server`, `spacy` en model, `unidic` |
| Cache | voice index pre-filled at build time (`phoonnx-voices update-cache`) |
| Port | `9666` |

## Configuration

The voice and synthesis options come from `mycroft.conf`. Mount one into the
container:

```yaml
# docker-compose.yml
    volumes:
      - phoonnx-cache:/home/ovos/.cache
      - ./mycroft.conf:/home/ovos/.config/mycroft/mycroft.conf:ro
```

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "<voice-id>",

      "ref_wav": "/ref/me.wav",
      "ref_text": "olá tudo bem com você",
      "ref_lang": "pt"
    }
  }
}
```

`ref_wav` / `ref_text` / `ref_lang` enable zero-shot [voice cloning](cloning.md) — the
text/lang are only needed by in-context engines (ZipVoice); d-vector voices clone from
`ref_wav` alone.

## Persistence

The `phoonnx-cache` named volume keeps the voice index and downloaded models across
restarts. Models download on first use of a voice; the pre-filled index means the
server itself starts without needing the network.

## Building locally

```bash
docker build -t phoonnx .
```

The [`docker` workflow](../.github/workflows/docker.yml) builds on every PR and
publishes to `ghcr.io/tigregotico/phoonnx` on pushes to `master` (`latest`), `dev`
(`dev`), and version tags.

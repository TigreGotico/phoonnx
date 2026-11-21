# Docker TTS server

<!-- agent-hint
covers: how to run the published phoonnx image as an OVOS TTS server, what the image holds, and how to configure and build it
source: Dockerfile, docker-compose.yml, docker-entrypoint.sh, .github/workflows/docker.yml
generated: no
verify: grep -n "FROM\|uv pip install\|update-cache\|EXPOSE" Dockerfile; grep -n "type=raw" .github/workflows/docker.yml
see: deployment.md, ovos_plugin.md, offline.md, installation.md
-->

This page shows how to run phoonnx as a containerized text-to-speech server. It is for operators who serve speech over HTTP.

phoonnx ships an image that runs it as an
[`ovos-tts-server`](https://github.com/OpenVoiceOS/ovos-tts-server). The image holds **every
optional dependency** (all language phonemizers and voice cloning), the **espeak-ng**
binary, and a **pre-filled voice index**, so it starts on a cold cache.

## Quick start

```bash
docker run -p 9666:9666 ghcr.io/tigregotico/phoonnx:dev
```

The `dev` tag is built from the `dev` branch. The image is about 1.4 GB to download.

Or, from a checkout of this repository, start it with the bundled compose file, which runs
the same image:

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
| System | `espeak-ng`, `libsndfile1`, `ffmpeg` |
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

`ref_wav`, `ref_text` and `ref_lang` enable zero-shot [voice cloning](cloning.md). Only
in-context engines (ZipVoice) need the text and the language. D-vector voices clone from
`ref_wav` alone.

## Persistence

The `phoonnx-cache` named volume keeps the voice index and downloaded models across
restarts. A model downloads on the first use of its voice. Because the index is
pre-filled, the server starts without the network.

To run this image as a public or long-lived service, see [deployment.md](deployment.md).
It covers the cache volume, prefetching voice weights before you serve traffic, health
checks, and sizing.

## Building locally

```bash
docker build -t phoonnx .
```

The [`docker` workflow](../.github/workflows/docker.yml) builds on every PR. It publishes
to `ghcr.io/tigregotico/phoonnx` with the tag `dev` on a push to `dev`, `latest` on a push to
`master`, and a version tag on a release tag. The pages use `:dev`, because the registry
holds `latest` only after a push to `master`.

---

[Docs index](index.md) · Previous: [OVOS plugin](ovos_plugin.md) · Next: [Deployment](deployment.md)

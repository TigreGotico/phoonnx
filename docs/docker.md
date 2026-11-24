# Docker TTS server

<!-- agent-hint
covers: how to run the published phoonnx image as an OVOS TTS server, what the image holds, and how to configure and build it
source: Dockerfile, docker-compose.yml, docker-entrypoint.sh, .github/workflows/docker.yml
generated: no
verify: .venv/bin/python -c "import re,sys; p=open('docs/docker.md').read(); d=open('Dockerfile').read(); w=open('.github/workflows/docker.yml').read(); q=chr(96); row=lambda k: re.findall(q+'([^'+q+']+)'+q, re.search('^[|] '+k+' [|](.*)$', p, re.M).group(1)); apt=[x for x in re.search('no-install-recommends(.*?)&&', d, re.S).group(1).split() if x!=chr(92)]; ok=[row('Base')==[d.split('FROM ')[1].split()[0]], row('System')==apt, row('Port')==re.findall('EXPOSE ([0-9]+)', d), all('value='+t+',' in w for t in row('Branch tags'))]; print(ok); sys.exit(0 if all(ok) else 1)"
see: deployment.md, ovos_plugin.md, offline.md, installation.md
-->

This page shows how to run phoonnx as a text-to-speech server in a container. It is for operators who serve speech over HTTP.

The phoonnx image runs [`ovos-tts-server`](https://github.com/OpenVoiceOS/ovos-tts-server) with the phoonnx plugin. The image holds the `all` extra, which installs every language phonemizer and the voice cloning dependencies. It also holds the espeak-ng binary and a filled voice index, so it starts with an empty cache.

## Run the server

1. Start the image:

   ```bash
   docker run -p 9666:9666 ghcr.io/tigregotico/phoonnx:dev
   ```

   The `dev` tag is built from the `dev` branch. From a checkout of this repository, `docker compose up -d` starts the same image with a cache volume and a health check.

2. Synthesize a sentence:

   ```bash
   curl "http://localhost:9666/synthesize/hello%20world" --output hello.wav
   ```

3. Check the server status:

   ```bash
   curl http://localhost:9666/status
   ```

The server also serves `/v2/synthesize?utterance=...`. The entrypoint starts `ovos-tts-server --engine ovos-tts-plugin-phoonnx --cache`, so the server keeps each synthesized sentence on disk. Arguments after the image name go to `ovos-tts-server`.

## Configure the voice

The voice and the synthesis options come from `mycroft.conf`. Mount one into the container. The compose file has this line as a comment:

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

`ref_wav`, `ref_text` and `ref_lang` turn on zero-shot [voice cloning](cloning.md). In-context engines, such as ZipVoice, need the text and the language. D-vector voices clone from `ref_wav` alone. The [OVOS plugin](ovos_plugin.md) page lists every option.

## Keep the cache

The `phoonnx-cache` named volume keeps the voice index, the downloaded models and the synthesis cache across restarts. A model downloads on the first use of its voice. The index is filled when the image is built, so the server starts with no network.

To run the image as a public or long-lived service, read [Deployment](deployment.md). It covers the cache volume, the download of voice weights before the server takes traffic, health checks and memory.

## Build the image

```bash
docker build -t phoonnx .
```

The [`docker` workflow](../.github/workflows/docker.yml) builds the image for each pull request that changes the `Dockerfile`, `.dockerignore`, `pyproject.toml` or the workflow. It publishes to `ghcr.io/tigregotico/phoonnx` on a push to `dev` or `master` and on a release tag. Each published image also gets a tag with the short commit sha.

## What the image holds

| Part | Value |
|---|---|
| Base | `python:3.11-slim` |
| System | `espeak-ng` `libsndfile1` `ffmpeg` `git` `build-essential` |
| Python | `phoonnx[all]`, `ovos-tts-server[mcp]`, the CPU build of `torch`, the spaCy `en_core_web_sm` model and the `unidic` dictionary |
| Cache | the voice index, filled at build time by `phoonnx-voices update-cache` |
| User | `ovos`, uid 1000 |
| Port | `9666` |
| Branch tags | `dev` from the dev branch, `latest` from the master branch |

---

[Docs index](index.md) · Previous: [OVOS plugin](ovos_plugin.md) · Next: [Deployment](deployment.md)
